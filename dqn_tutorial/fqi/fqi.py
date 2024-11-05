"""
Fitted Q-Iteration
as described in "Tree-based batch mode reinforcement learning"
by Ernst et al. and
"Neural fitted Q iteration" by Martin Riedmiller.
"""

from functools import partial
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sklearn import tree
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from dqn_tutorial.fqi import load_data


def create_model_input(
    obs: np.ndarray,
    actions: np.ndarray,
    features_extractor: Optional[PolynomialFeatures] = None,
) -> np.ndarray:
    """
    Concatenate observation (batch_size, n_features)
    and actions (batch_size, 1) along the feature axis.

    :param obs: A batch of observations.
    :param actions: A batch of actions.
    :param features_extractor: Optionally a preprocessor
        to extract features like PolynomialFeatures.
    :return: The input for the scikit-learn model
        (batch_size, n_features + 1)
    """
    # Concatenate the observations and actions
    # so we can predict qf(s_t, a_t)
    model_input = np.concatenate((obs, actions), axis=1)
    # Optionally: extract features from the input using preprocessor
    if features_extractor is not None:
        try:
            model_input = features_extractor.transform(model_input)
        except NotFittedError:
            # First interation: fit the features_extractor
            model_input = features_extractor.fit_transform(model_input)
    return model_input


def get_q_values(
    model: RegressorMixin,
    obs: np.ndarray,
    n_actions: int,
    features_extractor: Optional[PolynomialFeatures] = None,
) -> np.ndarray:
    """
    Retrieve the q-values for a set of observations (=states in the theory).
    qf(states, action) for all possible actions.

    :param model: Q-value estimator
    :param obs: A batch of observations
    :param n_actions: Number of discrete actions.
    :param features_extractor: Optionally a preprocessor
        to extract features like PolynomialFeatures.
    :return: The predicted q-values for the given observations
        (batch_size, n_actions)
    """
    batch_size = len(obs)
    q_values = np.zeros((batch_size, n_actions))
    # Predict q-value for each action
    for action_idx in range(n_actions):
        # Note: we should do one hot encoding if not using CartPole (n_actions > 2)
        # Create a vector of size batch_size for the current action
        actions = action_idx * np.ones((batch_size, 1))
        # Concatenate the observations and the actions to obtain
        # the input to the q-value estimator
        model_input = create_model_input(obs, actions, features_extractor)
        # Predict q-values for the given observation/action combination
        # shape: (batch_size, 1)
        predicted_q_values = model.predict(model_input)
        q_values[:, action_idx] = predicted_q_values

    return q_values


def evaluate(
    model: RegressorMixin,
    env: gym.Env,
    n_eval_episodes: int = 10,
    features_extractor: Optional[PolynomialFeatures] = None,
) -> None:
    episode_returns, episode_reward = [], 0.0
    total_episodes = 0
    done = False
    obs, _ = env.reset()
    assert isinstance(env.action_space, spaces.Discrete), "FQI only support discrete actions"

    while total_episodes < n_eval_episodes:
        # Retrieve the q-values for the current observation
        q_values = get_q_values(
            model,
            obs[np.newaxis, ...],
            int(env.action_space.n),
            features_extractor,
        )
        # Select the action that maximizes the q-value for each state
        best_action = int(np.argmax(q_values, axis=1).item())
        # Send the action to the env
        obs, reward, terminated, truncated, _ = env.step(best_action)
        episode_reward += float(reward)

        done = terminated or truncated
        if done:
            episode_returns.append(episode_reward)
            episode_reward = 0.0
            total_episodes += 1
            obs, _ = env.reset()
    print(f"Total reward = {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")


if __name__ == "__main__":  # pragma: no cover
    # Max number of iterations
    n_iterations = 50
    # How often do we evaluate the learned model
    eval_freq = 5
    # How many episodes to evaluate every eval-freq
    n_eval_episodes = 2
    # discount factor
    gamma = 0.99
    # For visualization: "human", "rgb_array" or None
    render_mode = "human"
    # Scikit learn model to use
    model_name = "knn"

    model_class = {
        "linear": LinearRegression,
        "mlp": partial(MLPRegressor, hidden_layer_sizes=(64, 64), early_stopping=True, n_iter_no_change=2),
        "tree": partial(tree.DecisionTreeRegressor),
        "forest": RandomForestRegressor,
        "knn": partial(KNeighborsRegressor, n_neighbors=30),
        "tree_boost": partial(GradientBoostingRegressor, n_estimators=50),
    }[model_name]

    # Optionally: extract features before feeding the input to the model
    # features_extractor = PolynomialFeatures(degree=2)
    features_extractor = None

    env_id = "CartPole-v1"
    output_filename = Path("data") / f"{env_id}_data.npz"
    # Create test environment
    env = gym.make(env_id, render_mode=render_mode)

    assert isinstance(env.action_space, spaces.Discrete), "FQI only support discrete actions"

    # Load saved transitions
    data = load_data(output_filename)

    # First iteration:
    # The target q-value is the reward obtained
    targets = data.rewards.copy()
    # Create input for current observations
    current_obs_input = create_model_input(data.observations, data.actions, features_extractor)
    # Fit the estimator for the current target
    model = model_class().fit(current_obs_input, targets)

    try:
        for iter_idx in range(n_iterations):
            # Construct TD(0) target
            # using current model and the next observations
            next_q_values = get_q_values(
                model,
                data.next_observations,
                n_actions=int(env.action_space.n),
                features_extractor=features_extractor,
            )
            # Follow-greedy policy: use the action with the highest q-value
            next_q_values = next_q_values.max(axis=1)
            # The new target is the reward + what our agent expect to get
            # if it follows a greedy policy (follow action with the highest q-value)
            targets = data.rewards + gamma * (1 - data.terminateds) * next_q_values
            # Update our q-value estimate with the current target
            model = model_class().fit(current_obs_input, targets)

            if (iter_idx + 1) % eval_freq == 0:
                print(f"Iter {iter_idx + 1}")
                print(f"Score: {model.score(current_obs_input, targets):.2f}")
                evaluate(model, env, n_eval_episodes, features_extractor)
    except KeyboardInterrupt:
        pass
