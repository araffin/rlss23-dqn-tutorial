from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from dqn_tutorial.dqn.collect_data import collect_one_step, linear_schedule
from dqn_tutorial.dqn.evaluation import evaluate_policy
from dqn_tutorial.dqn.q_network import QNetwork
from dqn_tutorial.dqn.replay_buffer import ReplayBuffer


def dqn_update(
    q_net: QNetwork,
    q_target_net: QNetwork,
    optimizer: th.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> None:
    """
    Perform one gradient step on the Q-network
    using the data from the replay buffer.

    :param q_net: The Q-network to update
    :param q_target_net: The target Q-network, to compute the td-target.
    :param optimizer: The optimizer to use
    :param replay_buffer: The replay buffer containing the transitions
    :param batch_size: The minibatch size, how many transitions to sample
    :param gamma: The discount factor
    """

    # Sample the replay buffer and convert them to PyTorch tensors
    replay_data = replay_buffer.sample(batch_size).to_torch()

    with th.no_grad():
        # Compute the Q-values for the next observations (batch_size, n_actions)
        # using the target network
        next_q_values = q_target_net(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        # (batch_size,)
        next_q_values, _ = next_q_values.max(dim=1)
        # If the episode is terminated, set the target to the reward
        should_bootstrap = th.logical_not(replay_data.terminateds)
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
    # Huber loss
    # loss = th.nn.functional.smooth_l1_loss(current_q_values, td_target)

    # Reset gradients
    optimizer.zero_grad()
    # Compute the gradients
    loss.backward()
    # Update the parameters of the q-network
    optimizer.step()


def run_dqn(
    env_id: str = "CartPole-v1",
    replay_buffer_size: int = 50_000,
    # How often do we copy the parameters from the Q-network to the target network
    target_network_update_interval: int = 1000,
    # Warmup phase
    learning_starts: int = 100,
    # Exploration schedule
    # (for the epsilon-greedy data collection)
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.01,
    exploration_fraction: float = 0.1,
    n_timesteps: int = 20_000,
    update_interval: int = 2,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    gamma: float = 0.99,
    n_hidden_units: int = 64,
    n_eval_episodes: int = 10,
    evaluation_interval: int = 1000,
    eval_exploration_rate: float = 0.0,
    seed: int = 2023,
    # device: Union[th.device, str] = "cpu",
    eval_render_mode: Optional[str] = None,  # "human", "rgb_array", None
) -> QNetwork:
    """
    Run Deep Q-Learning (DQN) on a given environment.
    (with a target network)

    :param env_id: Name of the environment
    :param replay_buffer_size: Max capacity of the replay buffer
    :param target_network_update_interval: How often do we copy the parameters
         to the target network
    :param learning_starts: Warmup phase to fill the replay buffer
        before starting the optimization.
    :param exploration_initial_eps: The initial exploration rate
    :param exploration_final_eps: The final exploration rate
    :param exploration_fraction: The fraction of the number of steps
        during which the exploration rate is annealed from
        initial_eps to final_eps.
        After this many steps, the exploration rate remains constant.
    :param n_timesteps: Number of timesteps in total
    :param update_interval: How often to update the Q-network
        (every update_interval steps)
    :param learning_rate: The learning rate to use for the optimizer
    :param batch_size: The minibatch size
    :param gamma: The discount factor
    :param n_hidden_units: Number of units for each hidden layer
        of the Q-Network.
    :param n_eval_episodes: The number of episodes to evaluate the policy on
    :param evaluation_interval: How often to evaluate the policy
    :param eval_exploration_rate: The exploration rate to use during evaluation
    :param seed: Random seed for the pseudo random generator
    :param eval_render_mode: The render mode to use for evaluation
    """
    # Set seed for reproducibility
    # Seed Numpy as PyTorch pseudo random generators
    # Seed Numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    th.manual_seed(seed)

    # Create the environment
    env = gym.make(env_id)
    # For highway env
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    env.action_space.seed(seed)

    # Create the evaluation environment
    eval_env = gym.make(env_id, render_mode=eval_render_mode)
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    # Create the q-network
    q_net = QNetwork(env.observation_space, env.action_space, n_hidden_units=n_hidden_units)
    # Create the target network
    q_target_net = QNetwork(env.observation_space, env.action_space, n_hidden_units=n_hidden_units)
    # Copy the parameters of the q-network to the target network
    q_target_net.load_state_dict(q_net.state_dict())

    # For flappy bird
    if env.observation_space.dtype == np.float64:
        q_net.double()
        q_target_net.double()

    # Create the optimizer, we only optimize the parameters of the q-network
    optimizer = th.optim.Adam(q_net.parameters(), lr=learning_rate)

    # Create the Replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, env.observation_space, env.action_space)
    # Reset the env
    obs, _ = env.reset(seed=seed)
    for current_step in range(1, n_timesteps + 1):
        # Update the current exploration schedule (update the value of epsilon)
        exploration_rate = linear_schedule(
            exploration_initial_eps,
            exploration_final_eps,
            current_step,
            int(exploration_fraction * n_timesteps),
        )
        # Do one step in the environment following an epsilon-greedy policy
        # and store the transition in the replay buffer
        obs = collect_one_step(
            env,
            q_net,
            replay_buffer,
            obs,
            exploration_rate=exploration_rate,
            verbose=0,
        )

        # Update the target network
        # by copying the parameters from the Q-network every target_network_update_interval steps
        if (current_step % target_network_update_interval) == 0:
            q_target_net.load_state_dict(q_net.state_dict())

        # Update the Q-network every update_interval steps
        # after learning_starts steps have passed (warmup phase)
        if (current_step % update_interval) == 0 and current_step > learning_starts:
            # Do one gradient step
            dqn_update(q_net, q_target_net, optimizer, replay_buffer, batch_size, gamma=gamma)

        if (current_step % evaluation_interval) == 0:
            print()
            print(f"Evaluation at step {current_step}:")
            print(f"exploration_rate={exploration_rate:.2f}")
            # Evaluate the current greedy policy (deterministic policy)
            evaluate_policy(eval_env, q_net, n_eval_episodes, eval_exploration_rate=eval_exploration_rate)
    return q_net


if __name__ == "__main__":  # pragma: no cover
    # Tuned hyperparameters from the RL Zoo3 of the Stable Baselines3 library
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    run_dqn(
        env_id="CartPole-v1",
        replay_buffer_size=100_000,
        # Note: you can remove the target network
        # by setting target_network_update_interval=1
        target_network_update_interval=10,
        learning_starts=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.04,
        exploration_fraction=0.1,
        n_timesteps=80_000,
        update_interval=2,
        learning_rate=1e-3,
        batch_size=64,
        gamma=0.99,
        n_eval_episodes=10,
        evaluation_interval=5000,
        # No exploration during evaluation
        # (deteministic policy)
        eval_exploration_rate=0.0,
        seed=2022,
    )

    # Same, for the LunarLander-v2 environment
    # run_dqn(
    #     env_id="LunarLander-v2",
    #     replay_buffer_size=50_000,
    #     # Note: you can remove the target network
    #     # by setting target_network_update_interval=1
    #     target_network_update_interval=250,
    #     learning_starts=100,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.1,
    #     exploration_fraction=0.12,
    #     n_timesteps=100_000,
    #     update_interval=2,
    #     learning_rate=6e-4,
    #     batch_size=128,
    #     gamma=0.99,
    #     n_eval_episodes=10,
    #     evaluation_interval=10000,
    #     # No exploration during evaluation
    #     # (deteministic policy)
    #     eval_exploration_rate=0.0,
    #     seed=2022,
    # )

    # Same, for the MountainCar-v0 environment
    # run_dqn(
    #     env_id="MountainCar-v0",
    #     replay_buffer_size=10_000,
    #     # Note: you can remove the target network
    #     # by setting target_network_update_interval=1
    #     target_network_update_interval=600,
    #     learning_starts=1000,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.07,
    #     exploration_fraction=0.15,
    #     n_timesteps=200_000,
    #     update_interval=16,
    #     learning_rate=4e-3,
    #     batch_size=128,
    #     gamma=0.98,
    #     n_eval_episodes=10,
    #     evaluation_interval=20000,
    #     # With noise during evaluation
    #     eval_exploration_rate=0.07,
    #     seed=2023,
    # )

    # For highway-fast env
    # import highway_env  noqa: F401
    #
    # run_dqn(
    #     env_id="highway-fast-v0",
    #     n_timesteps=20_000,
    #     # policy_kwargs=dict(net_arch=[256, 256]),
    #     learning_rate=5e-4,
    #     replay_buffer_size=15_000,
    #     learning_starts=200,
    #     batch_size=32,
    #     gamma=0.8,
    #     update_interval=4,
    #     target_network_update_interval=50,
    #     n_eval_episodes=2,
    #     evaluation_interval=500,
    #     # No noise during evaluation
    #     eval_exploration_rate=0.0,
    #     seed=2023,
    #     eval_render_mode="human",
    # )

    # import flappy_bird_gymnasium
    #
    # run_dqn(
    #     env_id="FlappyBird-v0",
    #     replay_buffer_size=100_000,
    #     # Note: you can remove the target network
    #     # by setting target_network_update_interval=1
    #     target_network_update_interval=250,
    #     learning_starts=10_000,
    #     exploration_initial_eps=1.0,
    #     exploration_final_eps=0.03,
    #     exploration_fraction=0.1,
    #     n_timesteps=500_000,
    #     update_interval=4,
    #     learning_rate=1e-3,
    #     batch_size=128,
    #     gamma=0.98,
    #     n_eval_episodes=2,
    #     evaluation_interval=50000,
    #     n_hidden_units=256,
    #     # No exploration during evaluation
    #     # (deteministic policy)
    #     eval_exploration_rate=0.0,
    #     seed=2023,
    #     eval_render_mode="human",
    # )
