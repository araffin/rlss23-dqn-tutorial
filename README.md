# Reinforcement Learning Summer School 2023 - DQN Tutorial

Website: https://rlsummerschool.com/


## Tasks

### Fitted Q-Iteration (FQI) - 30 minutes

1. `collect_data()` function (Gym API, 5 minutes)
2. Fitted Q-Iteration (FQI) (25 minutes):
  - `get_q_values()` function
  - `evaluate()` function
  - update rule using TD(0) target
  - play with different models/features_extractor

### Deep Q-Network (DQN) - 1h30

#### Mix offline and online data

3. Create the `ReplayBuffer` class
4. Create the Q-network
5. Epsilon-greedy data collection (re-using FQI)
6. Write DQN update rule (no target network)

#### With target network

6. Add DQN target network and periodic copy

Explore different value for the target update,
use soft update instead of hard-copy.

Compare to SB3/SBX results.

Bonus: CNN and learn on Pong
+ learn discretized version of upkie?

#### DQN Extensions
- Prioritized-Experience Replay (PER)?
- Double DQN (DDQN)
- noisy net
