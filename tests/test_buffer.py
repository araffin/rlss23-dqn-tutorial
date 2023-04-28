import gymnasium as gym
from gymnasium import spaces

from dqn_tutorial.dqn import ReplayBuffer


def test_buffer() -> None:
    env = gym.make("CartPole-v1")
    assert isinstance(env.observation_space, spaces.Box)
    assert isinstance(env.action_space, spaces.Discrete)
    buffer = ReplayBuffer(1000, env.observation_space, env.action_space)
    obs, _ = env.reset()
    # Fill the buffer
    for _ in range(500):
        action = int(env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.store_transition(obs, next_obs, action, float(reward), terminated)
        # Update current observation
        obs = next_obs

        done = terminated or truncated
        if done:
            obs, _ = env.reset()

    assert not buffer.is_full
    assert buffer.current_idx == 500
    samples = buffer.sample(batch_size=10)
    assert len(samples.observations) == 10
    assert samples.actions.shape == (10, 1)

    # Fill the buffer completely
    for _ in range(1000):
        action = int(env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.store_transition(obs, next_obs, action, float(reward), terminated)
        # Update current observation
        obs = next_obs

        done = terminated or truncated
        if done:
            obs, _ = env.reset()

    assert buffer.is_full
    # We did a full loop
    assert buffer.current_idx == 500
    # Check sampling with replacement
    samples = buffer.sample(batch_size=1001)
    assert len(samples.observations) == 1001
    assert samples.actions.shape == (1001, 1)
