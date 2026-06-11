"""
Solutions to the FQI notebook to bypass exercises.
"""

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from sklearn.base import RegressorMixin


def get_q_values(
    model: RegressorMixin,
    obs: np.ndarray,
    n_actions: int,
) -> np.ndarray:
    """
    Retrieve the q-values for a set of observations.
    qf(q_t, action) for all possible actions.

    :param model: Q-value estimator
    :param obs: A batch of observations
    :param n_actions: Number of discrete actions.
    :return: The predicted q-values for the given observations
        (batch_size, n_actions)
    """
    batch_size = len(obs)
    q_values = np.zeros((batch_size, n_actions))

    ### YOUR CODE HERE
    # TODO: for every possible actions a:
    # 1. Create the regression model input $(s, a)$ for the action a
    # and states s (here a batch of observations)
    # 2. Predict the q-values for the batch of states
    # 3. Update q-values array for the current action a

    # Predict q-value for each action
    for action_idx in range(n_actions):
        # Note: we should do one hot encoding if not using CartPole (n_actions > 2)
        # Create a vector of size (batch_size, 1) for the current action
        # This allows to do batch prediction for all the provided observations
        actions = action_idx * np.ones((batch_size, 1))
        # Concatenate the observations and the actions to obtain
        # the input to the q-value estimator
        # you can use `np.concatenate()`
        model_input = np.concatenate((obs, actions), axis=1)
        # Predict q-values for the given observation/action combination
        # shape: (batch_size, 1)
        predicted_q_values = model.predict(model_input)
        # Update the q-values array for the current action
        q_values[:, action_idx] = predicted_q_values

    ### END OF YOUR CODE

    return q_values


def evaluate(
    model: RegressorMixin,
    env: gym.Env,
    n_eval_episodes: int = 10,
    video_name: str | None = None,
    video_path: str | None = None,
) -> None:
    episode_returns, episode_reward = [], 0.0
    total_episodes = 0
    done = False
    video_path = video_path or "../logs/videos"

    # Setup video recorder
    if video_name is not None and env.render_mode == "rgb_array":
        os.makedirs(video_path, exist_ok=True)

        # New gym recorder always wants to cut video into episodes,
        # set video length big enough but not to inf (will cut into episodes)
        env = RecordVideo(env, video_path, step_trigger=lambda _: False, video_length=100_000)
        env.start_recording(video_name)

    obs, _ = env.reset()
    assert isinstance(env.action_space, spaces.Discrete), "FQI only support discrete actions"
    n_actions = int(env.action_space.n)

    while total_episodes < n_eval_episodes:
        ### YOUR CODE HERE

        # Retrieve the q-values for the current observation
        # you need to re-use `get_q_values()`
        # Note: you need to add a batch dimension to the observation
        # you can use `obs[np.newaxis, ...]` for that: (obs_dim,) -> (batch_size=1, obs_dim)
        q_values = get_q_values(
            model,
            obs[np.newaxis, ...],
            n_actions,
        )
        # Select the action that maximizes the q-value for each state
        # Don't forget to remove the batch dimension, you can `.item()` for that
        best_action = int(np.argmax(q_values, axis=1).item())

        # Send the action to the env
        obs, reward, terminated, truncated, _ = env.step(best_action)

        ### END OF YOUR CODE

        episode_reward += float(reward)

        done = terminated or truncated
        if done:
            episode_returns.append(episode_reward)
            episode_reward = 0.0
            total_episodes += 1
            obs, _ = env.reset()

    if isinstance(env, RecordVideo):
        print(f"Saving video to {video_path}/{video_name}")
        env.close()

    print(f"Total reward = {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
