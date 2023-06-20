from dataclasses import dataclass

import numpy as np
import torch as th
from gymnasium import spaces


@dataclass
class TorchReplayBufferSamples:
    observations: th.Tensor
    next_observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    terminateds: th.Tensor


@dataclass
class ReplayBufferSamples:
    """
    A dataclass containing transitions from the replay buffer.
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminateds: np.ndarray

    def to_torch(self, device: str = "cpu") -> TorchReplayBufferSamples:
        """
        Convert the samples to PyTorch tensors.

        :param device: PyTorch device
        :return: Samples as PyTorch tensors
        """
        return TorchReplayBufferSamples(
            observations=th.as_tensor(self.observations, device=device),
            next_observations=th.as_tensor(self.next_observations, device=device),
            actions=th.as_tensor(self.actions, device=device),
            rewards=th.as_tensor(self.rewards, device=device),
            terminateds=th.as_tensor(self.terminateds, device=device),
        )


class ReplayBuffer:
    """
    A simple replay buffer class to store and sample transitions.

    :param buffer_size: Max number of transitions to store
    :param observation_space: Observation space of the env,
        contains information about the observation type and shape.
    :param action_space: Action space of the env,
        contains information about the number of actions.
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
        self.terminateds = np.zeros((buffer_size,), dtype=bool)

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
        self.terminateds[self.current_idx] = terminated
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
            self.terminateds[batch_indices],
        )
