from typing import Type

import torch as th
import torch.nn as nn
from gymnasium import spaces


class QNetwork(nn.Module):
    """
    A Q-Network for the DQN algorithm
    to estimate the q-value for a given observation.

    :param observation_space: Observation space of the env,
        contains information about the observation type and shape.
    :param action_space: Action space of the env,
        contains information about the number of actions.
    :param n_hidden_units: Number of units for each hidden layer.
    :param activation_fn: Activation function (ReLU by default)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        n_hidden_units: int = 64,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        # Assume 1d space
        obs_dim = observation_space.shape[0]
        # Retrieve the number of discrete actions
        n_actions = int(action_space.n)
        # Create the q network (2 fully connected hidden layers)
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_hidden_units),
            activation_fn(),
            nn.Linear(n_hidden_units, n_actions),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :param observations: A batch of observation (batch_size, obs_dim)
        :return: The Q-values for the given observations
            for all the action (batch_size, n_actions)
        """
        return self.q_net(observations)
