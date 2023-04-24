from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


@dataclass
class OfflineData:
    """
    A class to store transitions.
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray


def collect_data(env_id: str, n_steps: int = 50_000) -> OfflineData:
    """
    Collect transitions using a random agent (sample action randomly).

    :param env_id: The name of the environment.
    :param n_steps: Number of steps to perform in the env.
    :return: The collected transitions.
    """
    # Create the Gym env
    env = gym.make(env_id)

    assert isinstance(env.observation_space, gym.spaces.Box)
    observations = np.zeros((n_steps, *env.observation_space.shape))
    next_observations = np.zeros((n_steps, *env.observation_space.shape))
    # Discrete actions
    actions = np.zeros((n_steps, 1))
    rewards = np.zeros((n_steps, 1))
    terminateds = np.zeros((n_steps, 1))

    done = False
    obs, _ = env.reset()

    for idx in range(n_steps):
        # Sample a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info_ = env.step(action)

        # Store the transition
        observations[idx, :] = obs
        next_observations[idx, :] = next_obs
        actions[idx, :] = action
        rewards[idx, :] = reward
        # Only record true termination (timeout will never happened with real data)
        terminateds[idx, :] = terminated
        obs = next_obs
        # Check if the episode is over
        done = terminated or truncated

        if done:
            obs, _ = env.reset()
    return OfflineData(
        observations,
        next_observations,
        actions,
        rewards,
        terminateds,
    )


def save_data(data: OfflineData, path: Path) -> None:
    """
    Save the collected transitions to a file (npz archive).

    :param data: Collected data (transitions)
    :param path: Where to save the data (without the extension)
    """
    print(f"Saving to {path}.npz")

    np.savez(
        path,
        **dict(
            obs=data.observations,
            next_obs=data.next_observations,
            rewards=data.rewards,
            actions=data.actions,
            terminated=data.terminateds,
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    env_id = "CartPole-v1"
    output_filename = Path("data") / f"{env_id}_data"
    # Create folder if it doesn't exist
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    # Collect data
    data = collect_data(env_id, n_steps=50_000)
    # Save collected data using numpy
    save_data(data, output_filename)
