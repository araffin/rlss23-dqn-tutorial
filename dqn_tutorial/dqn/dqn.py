import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from dqn_tutorial.dqn.collect_data import collect_one_step, epsilon_greedy_action_selection, linear_schedule
from dqn_tutorial.dqn.q_network import QNetwork
from dqn_tutorial.dqn.replay_buffer import ReplayBuffer


def dqn_update(
    q_net: QNetwork,
    optimizer: th.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> None:
    """
    Perform one gradient step on the Q-network
    using the data from the replay buffer.

    :param q_net: The Q-network to update
    :param optimizer: The optimizer to use
    :param replay_buffer: The replay buffer containing the transitions
    :param batch_size: The minibatch size, how many transitions to sample
    :param gamma: The discount factor
    """

    # Sample the replay buffer and convert them to PyTorch tensors
    replay_data = replay_buffer.sample(batch_size).to_torch()

    with th.no_grad():
        # Compute the Q-values for the next observations (batch_size, n_actions)
        next_q_values = q_net(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        # (batch_size,)
        next_q_values, _ = next_q_values.max(dim=1)
        # If the episode is terminated, set the target to the reward
        should_bootstrap = th.logical_not(replay_data.terminated)
        # 1-step TD target
        td_target = replay_data.rewards + gamma * next_q_values * should_bootstrap

    # Get current Q-values estimates for the replay_data (batch_size, n_actions)
    q_values = q_net(replay_data.observations)
    # Select the Q-values corresponding to the actions that were selected
    # during data collection
    current_q_values = th.gather(q_values, dim=1, index=replay_data.actions)
    # Reshape from (batch_size, 1) to (batch_size,) to avoid broadcast error
    current_q_values = current_q_values.squeeze(dim=1)

    # Check for any shape/broadcast error
    # Current q-values must have the same shape as the TD target
    assert current_q_values.shape == (batch_size,), f"{current_q_values.shape} != {(batch_size,)}"
    assert current_q_values.shape == td_target.shape, f"{current_q_values.shape} != {td_target.shape}"

    # Compute the Mean Squared Error (MSE) loss
    # Optionally, one can use a Huber loss instead of the MSE loss
    loss = ((current_q_values - td_target) ** 2).mean()
    # Reset gradients
    optimizer.zero_grad()
    # Compute the gradients
    loss.backward()
    # Update the parameters of the q-network
    optimizer.step()


def evaluate_policy(eval_env: gym.Env, q_net: QNetwork, n_eval_episodes: int, eval_exploration_rate: float = 0.0) -> None:
    """
    Evaluate the policy by computing the average episode reward
    over n_eval_episodes episodes.

    :param eval_env: The environment to evaluate the policy on
    :param q_net: The Q-network to evaluate
    :param n_eval_episodes: The number of episodes to evaluate the policy on
    :param eval_exploration_rate: The exploration rate to use during evaluation
    """
    assert isinstance(eval_env.action_space, spaces.Discrete)

    episode_returns = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # Select the action according to the policy
            action = epsilon_greedy_action_selection(
                q_net,
                obs,
                exploration_rate=eval_exploration_rate,
                action_space=eval_env.action_space,
            )
            # Render
            if eval_env.render_mode is not None:
                eval_env.render()
            # Do one step in the environment
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)

            done = terminated or truncated
        # Store the episode reward
        episode_returns.append(total_reward)
    # Print mean and std of the episode rewards
    print(f"Mean episode reward: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")


if __name__ == "__main__":
    # Hyperparameters
    # Name of the environment
    env_id = "CartPole-v1"
    # Max capacity of the replay buffer
    replay_buffer_size = 50_000
    # Exploration schedule
    # (for the epsilon-greedy data collection)
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.01
    # Number of timesteps in total
    n_timesteps = 20_000
    # How often do we update the q-network
    # (every update_interval steps)
    update_interval = 2
    # Learning rate for the gradient descent
    learning_rate = 3e-4
    # Minibatch size
    batch_size = 64
    # discount factor
    gamma = 0.99
    # Number of episodes to evaluate the policy
    n_eval_episodes = 10
    # How often do we evaluate the policy
    evaluation_interval = 5_000
    eval_render_mode = None  # "human"
    # Random seed for the pseudo random generator
    seed = 2023
    print(f"Seed = {seed} - {env_id}")

    # Seed Numpy as PyTorch pseudo random generators
    # Seed Numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    # Create the environment
    env = gym.make(env_id)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    env.action_space.seed(seed)

    # Create the evaluation environment
    eval_env = gym.make(env_id, render_mode=eval_render_mode)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    # Create the q-network
    q_net = QNetwork(env.observation_space, env.action_space)
    # Create the optimizer
    optimizer = th.optim.Adam(q_net.parameters(), lr=learning_rate)

    # Create the Replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space)
    # Reset the env
    obs, _ = env.reset(seed=seed)
    for current_step in range(1, n_timesteps + 1):
        # Update the current exploration schedule
        # (update the value of epsilon)
        exploration_rate = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            current_step,
            n_timesteps,
        )
        # Do one step in the environment following an epsilon-greedy policy
        # and store the transition in the replay buffer
        obs = collect_one_step(env, q_net, replay_buffer, obs, exploration_rate=exploration_rate)
        if (current_step % update_interval) == 0:
            # Do one gradient step
            dqn_update(q_net, optimizer, replay_buffer, batch_size, gamma=gamma)

        if (current_step % evaluation_interval) == 0:
            print()
            print(f"Evaluation at step {current_step}:")
            # Evaluate the current greedy policy (deterministic policy)
            evaluate_policy(eval_env, q_net, n_eval_episodes, eval_exploration_rate=0.0)
