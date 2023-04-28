import gymnasium as gym
import numpy as np
import torch as th

from dqn_tutorial.dqn import QNetwork, ReplayBuffer, collect_one_step


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


def test_collect_data():
    env = gym.make("CartPole-v1")
    q_net = QNetwork(env.observation_space, env.action_space)
    buffer = ReplayBuffer(2000, env.observation_space, env.action_space)
    obs, _ = env.reset()
    for _ in range(1000):
        obs = collect_one_step(env, q_net, buffer, obs, exploration_rate=0.1)
    assert buffer.current_idx == 1000
    # Collect more data
    for _ in range(1000):
        obs = collect_one_step(env, q_net, buffer, obs, exploration_rate=0.1)
    # Buffer is full
    assert buffer.current_idx == 0
    assert buffer.is_full
    # Linear schedule
    exploration_initial_eps = 1.0
    exploration_final_eps = 0.01
    exploration_rate = exploration_initial_eps
    n_steps = 100
    for step in range(n_steps + 1):
        progress = step / n_steps
        exploration_rate = exploration_initial_eps + progress * (exploration_final_eps - exploration_initial_eps)
        if step == 0:
            assert exploration_rate == exploration_initial_eps

        obs = collect_one_step(env, q_net, buffer, obs, exploration_rate=exploration_rate)

    assert np.allclose(exploration_rate, exploration_final_eps)
