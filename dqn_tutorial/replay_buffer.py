from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


@dataclass
class ReplayBufferSamples:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    """
    A simple replay buffer class to store and sample transitions.

    :param buffer_size: Max number of transitions to store
    :param observation_space: Observation space of the env
    :param action_space: Action space of the env
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
    ) -> None:
        self.current_idx = 0
        self.buffer_size = buffer_size
        self.is_full = False
        self.observation_space = observation_space
        self.action_space = action_space
        # Create the different buffers
        self.observations = np.zeros((buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        # The action is an integer
        action_dim = 1
        self.actions = np.zeros((buffer_size, action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.terminated = np.zeros((buffer_size,), dtype=bool)

    def store_transition(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
    ) -> None:
        """
        Store one transition in the buffer.

        :param obs: Current observation
        :param next_obs: Next observation
        :param action: Action taken for the current observation
        :param reward: Reward received after taking the action
        :param terminated: Whether it is the end of an episode or not
            (discarding episode truncation like timeout)
        """
        # Update the buffers
        self.observations[self.current_idx] = obs
        self.next_observations[self.current_idx] = next_obs
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.terminated[self.current_idx] = terminated
        # Update the pointer, this is a ring buffer, we start from zero again when the buffer is full
        self.current_idx += 1
        if self.current_idx == self.buffer_size:
            self.is_full = True
            self.current_idx = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample with replacement `batch_size` transitions from the buffer.

        :param batch_size: How many transitions to sample.
        :return: Samples from the replay buffer
        """
        upper_bound = self.buffer_size if self.is_full else self.current_idx
        batch_indices = np.random.randint(0, upper_bound, size=batch_size)
        return ReplayBufferSamples(
            self.observations[batch_indices],
            self.next_observations[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.terminated[batch_indices],
        )
