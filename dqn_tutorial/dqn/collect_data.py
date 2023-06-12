import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from dqn_tutorial.dqn.q_network import QNetwork
from dqn_tutorial.dqn.replay_buffer import ReplayBuffer


def epsilon_greedy_action_selection(
    q_net: QNetwork,
    observation: np.ndarray,
    exploration_rate: float,
    action_space: spaces.Discrete,
    device: str = "cpu",
) -> int:
    """
    Select an action according to an espilon-greedy policy:
    with a probability of epsilon (`exploration_rate`),
    sample a random action, otherwise follow the best known action
    according to the q-value.

    :param observation: A single observation.
    :param q_net: Q-network for estimating the q value
    :param exploration_rate: Current rate of exploration (in [0, 1], 0 means no exploration),
        probability to select a random action,
        this is "epsilon".
    :param action_space: Action space of the env,
        contains information about the number of actions.
    :param device: PyTorch device
    :return: An action selected according to the epsilon-greedy policy.
    """
    if np.random.rand() < exploration_rate:
        # Random action
        action = int(action_space.sample())
    else:
        # Greedy action
        with th.no_grad():
            # Convert to PyTorch and add a batch dimension
            obs_tensor = th.as_tensor(observation[np.newaxis, ...], device=device)
            # Compute q values for all actions
            q_values = q_net(obs_tensor)
            # Greedy policy: select action with the highest q value
            action = q_values.argmax().item()
    return action


def collect_one_step(
    env: gym.Env,
    q_net: QNetwork,
    replay_buffer: ReplayBuffer,
    obs: np.ndarray,
    exploration_rate: float = 0.1,
    verbose: int = 0,
) -> np.ndarray:
    """
    Collect one transition and fill the replay buffer following an epsilon greedy policy.

    :param env: The environment object.
    :param q_net: Q-network for estimating the q value
    :param replay_buffer: Replay buffer to store the new transitions.
    :param obs: The current observation.
    :param exploration_rate: Current rate of exploration (in [0, 1], 0 means no exploration),
        probability to select a random action,
        this is "epsilon".
    :param verbose: The verbosity level (1 to print some info).
    :return: The last observation (important when collecting data multiple times).
    """
    assert isinstance(env.action_space, spaces.Discrete)

    action = epsilon_greedy_action_selection(q_net, obs, exploration_rate, env.action_space)
    next_obs, reward, terminated, truncated, info = env.step(action)
    replay_buffer.store_transition(obs, next_obs, action, float(reward), terminated)
    # Update current observation
    obs = next_obs

    if "episode" in info and verbose >= 1:
        print(f"Episode return={float(info['episode']['r']):.2f} length={int(info['episode']['l'])}")

    done = terminated or truncated
    if done:
        # Don't forget to reset the env at the end of an episode
        obs, _ = env.reset()
    return obs


def linear_schedule(initial_value: float, final_value: float, current_step: int, max_steps: int) -> float:
    """
    Linear schedule for the exploration rate (epsilon).
    Note: we clip the value so the schedule is constant after reaching the final value
    at `max_steps`.

    :param initial_value: Initial value of the schedule.
    :param final_value: Final value of the schedule.
    :param current_step: Current step of the schedule.
    :param max_steps: Maximum number of steps of the schedule.
    :return: The current value of the schedule.
    """
    # Compute current progress (in [0, 1], 0 being the start)
    progress = current_step / max_steps
    # Clip the progress so the schedule is constant after reaching the final value
    progress = min(progress, 1.0)
    return initial_value + progress * (final_value - initial_value)
