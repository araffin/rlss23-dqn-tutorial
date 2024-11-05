import warnings
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

    gym_v1 = False
except ImportError:
    from gymnasium.wrappers import RecordVideo

    gym_v1 = True

from dqn_tutorial.dqn.collect_data import epsilon_greedy_action_selection
from dqn_tutorial.dqn.q_network import QNetwork


def evaluate_policy(
    eval_env: gym.Env,
    q_net: QNetwork,
    n_eval_episodes: int,
    eval_exploration_rate: float = 0.0,
    video_name: Optional[str] = None,
) -> None:
    """
    Evaluate the policy by computing the average episode reward
    over n_eval_episodes episodes.

    :param eval_env: The environment to evaluate the policy on
    :param q_net: The Q-network to evaluate
    :param n_eval_episodes: The number of episodes to evaluate the policy on
    :param eval_exploration_rate: The exploration rate to use during evaluation
    :param video_name: When set, the filename of the video to record.
    """
    # Setup video recorder
    video_recorder = None
    if video_name is not None and eval_env.render_mode == "rgb_array":
        video_path = Path(__file__).parent.parent.parent / "logs" / "videos" / video_name
        video_path.parent.mkdir(parents=True, exist_ok=True)

        if gym_v1:
            # New gym recorder always wants to cut video into episodes,
            # set video length big enough but not to inf (will cut into episodes)
            # Silence warnings when the folder already exists
            warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")
            eval_env = RecordVideo(eval_env, str(video_path.parent), step_trigger=lambda _: False, video_length=100_000)
            eval_env.start_recording(video_name)
        else:
            video_recorder = VideoRecorder(
                env=eval_env,
                base_path=str(video_path),
            )

    assert isinstance(eval_env.action_space, spaces.Discrete)

    episode_returns = []
    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            # Record video
            if video_recorder is not None:
                video_recorder.capture_frame()

            # Select the action according to the policy
            action = epsilon_greedy_action_selection(
                q_net,
                obs,
                exploration_rate=eval_exploration_rate,
                action_space=eval_env.action_space,
            )
            # Render
            if eval_env.render_mode is not None:  # pragma: no cover
                eval_env.render()
            # Do one step in the environment
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += float(reward)

            done = terminated or truncated
        # Store the episode reward
        episode_returns.append(total_reward)

    if video_recorder is not None:
        print(f"Saving video to {video_recorder.path}")
        video_recorder.close()
    elif isinstance(eval_env, RecordVideo):
        print(f"Saving video to {video_path}.mp4")
    eval_env.close()

    # Print mean and std of the episode rewards
    print(f"Mean episode reward: {np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}")
