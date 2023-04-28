import gymnasium as gym
import numpy as np
import torch as th

from dqn_tutorial.dqn import QNetwork


def test_q_net():
    env = gym.make("CartPole-v1")
    q_net = QNetwork(env.observation_space, env.action_space)
    obs, _ = env.reset()

    with th.no_grad():
        obs_tensor = th.as_tensor(obs[np.newaxis, ...])
        q_values = q_net(obs_tensor)
        assert q_values.shape == (1, 2)
        best_action = q_values.argmax().item()
        assert isinstance(best_action, int)
