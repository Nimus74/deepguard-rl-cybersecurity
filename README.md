# 🔐 DeepGuard — Cyber Security for Healthcare using Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![gymnasium](https://img.shields.io/badge/gymnasium-0.26.3-black)](https://gymnasium.farama.org/)
[![gym-idsgame](https://img.shields.io/badge/gym--idsgame-1.0.12-green)](https://github.com/Limmen/gym-idsgame)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> Simulation of attack and defense scenarios in healthcare computer networks using Reinforcement Learning algorithms. Project developed for **DeepGuard Inc.** in the field of cybersecurity for the healthcare sector.

![Project Overview](results/comparison_reward.png)

---

## Project Summary

This project investigates the use of **Reinforcement Learning** to simulate **cyber‑defense strategies in healthcare networks**.

Two approaches are compared:

- **SARSA** — a tabular Reinforcement Learning algorithm  
- **Double Deep Q‑Network (DDQN)** — a Deep Reinforcement Learning method

Experiments are conducted in the **gym-idsgame** environment under different attack strategies and environment complexities.

**Key insight:** deterministic attackers are easier to learn than stochastic ones, while deep reinforcement learning scales better than tabular approaches as the complexity of the environment increases.

---

## Experimental Setup

The experiments were performed using the `gym-idsgame` environment, which simulates attacker–defender interactions in simplified network infrastructures.

### Environments

The following configurations were used:

- `random_attack-v0`
- `maximal_attack-v0`
- `random_attack-v3`

The **v3 environment** introduces a larger observation and action space, enabling analysis of algorithm scalability with increasing complexity.

### Training

Training configuration:

**SARSA**
- 10,000 episodes on v0
- 20,000 episodes on v3
- ε‑greedy exploration with exponential decay

**DDQN**
- Replay buffer: 50,000 transitions
- Batch size: 64
- Target network update: every 500 steps
- Neural network: 2 hidden layers (128 units each)

### Evaluation

Evaluation was performed every **500 training episodes** using a **greedy policy** over **100 evaluation episodes**.

### Hardware

Experiments were executed on **Google Colab CPU runtime**.



---

## 📋 Table of Contents

- [Context and Objectives](#context-and-objectives)
- [Simulation Environment](#simulation-environment)
- [Implemented Algorithms](#implemented-algorithms)
  - [SARSA](#sarsa--section-1)
  - [DDQN](#ddqn--section-2)
- [Project Architecture](#project-architecture)
- [Results](#results)
- [Critical Analysis](#critical-analysis)
- [Installation and Usage](#installation-and-usage)
- [Applied Technical Fixes](#applied-technical-fixes)
- [References](#references)

---

## Context and Objectives

Healthcare computer networks store sensitive patient data and must ensure regulatory compliance (GDPR, HIPAA). The growing sophistication of cyber attacks makes adaptive, learning-capable defense systems necessary.

This project implements an **attack/defense simulation** system based on Reinforcement Learning, with the following objectives:

| Objective | Description |
|---|---|
| **Adaptive Defense** | Train defensive agents in simulated scenarios to develop robust strategies |
| **Vulnerability Identification** | Simulate attacks to detect weaknesses before they are exploited |
| **Resource Optimization** | Automate simulation to reduce manual workload |
| **Algorithm Benchmarking** | Compare tabular (SARSA) and deep (DDQN) approaches on the same scenario |

---

## Simulation Environment

The environment used is [`gym-idsgame`](https://github.com/Limmen/gym-idsgame) — an abstract Markov Game for OpenAI Gym that simulates attack/defense interactions on computer networks.

### Game Model

`gym-idsgame` implements a **two-agent Markov Game** where attacker and defender face each other in a simulated network:

```
┌──────────────────────────────────────────────────────────────────────────┐
│             SIMULATED NETWORK — base configuration (v0)                   │
│                                                                          │
│  [Attacker] ──attacks──▶ [Server] ◀──defends── [Defender (RL agent)]      │
│                                                                          │
│  Layers: 1  │  Server/layer: 1  │  Attack types: 10                       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Configurations Used

| Environment | Agent Role | Opponent | Notes |
|---|---|---|---|
| `idsgame-random_attack-v0` | Defender | Random policy | Base configuration |
| `idsgame-maximal_attack-v0` | Defender | Maximal policy (deterministic) | Deterministic and aggressive opponent; its regularity can make it more predictable for the defender than a stochastic attacker |
| `idsgame-random_attack-v3` | Defender | Random policy | Larger observation/action space (Section 4) |

### Detected Technical Features

```
obs_space.shape   (declared)   : (1, 11)
obs.flatten()     (actual)     : (33,)   ← includes info from both agents
N_ACTIONS                      : 30
step() input                   : (attacker_action, defender_action)  ← always tuple
step() output reward           : (att_reward, def_reward)            ← always tuple
info                           : {'moved': bool}  ← attack_success NOT exposed
```

> ⚠️ **API Note:** `gym-idsgame` maintains the two-agent interface even in single-agent variants. Our agent only controls `defender_action`; `attacker_action=0` is a placeholder ignored by the environment.

> ⚠️ **hack_rate Note:** `gym-idsgame` does not expose `info['attack_success']`. The hack_rate is calculated at episode level: an episode is considered compromised if the defender receives at least one negative reward during the episode (`reward < 0`).

---

## Implemented Algorithms

### SARSA — Section 1

**SARSA** (State-Action-Reward-State-Action) is an **on-policy** Temporal Difference learning algorithm.

#### Why SARSA for random_attack?

In an environment with a stochastic opponent, SARSA is preferable to Q-learning because:
- It learns the value of the policy it **is actually executing** (including exploration)
- Produces more **conservative and robust** estimates under uncertainty
- Avoids overestimating states that are reached in practice with a policy that has not yet converged

#### Update rule

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \cdot Q(s', a') - Q(s,a) \right]$$

where $a'$ is the action chosen by the **same** ε-greedy policy from $s'$ — not the hypothetical maximum as in Q-learning.

#### Implementation

```python
# Q-table as defaultdict: unvisited states → zeros automatically
Q = defaultdict(lambda: np.zeros(N_ACTIONS))

# State key: exact serialization without manual discretization
state_key = obs.flatten().tobytes()

# SARSA loop: a' chosen BEFORE update (on-policy)
a = epsilon_greedy(Q, s, epsilon)          # initial choice
while not done:
    obs_next, reward, done = env_step(env, a)
    a_next = epsilon_greedy(Q, s_next, epsilon)   # same policy
    Q[s][a] += alpha * (reward + gamma * Q[s_next][a_next] - Q[s][a])
    s, a = s_next, a_next
```

#### SARSA Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `ALPHA` | 0.001 | Conservative learning rate for dense Q-table |
| `GAMMA` | 0.99 | Long time horizon |
| `EPSILON_START` | 1.0 | Initial full exploration |
| `EPSILON_MIN` | 0.01 | Guaranteed minimum threshold |
| `EPSILON_DECAY` | 0.9995 | Slow decay: ε≈0.10 at ep 4600, ε≈0.01 at ep 9000 |

---

### DDQN — Section 2

**Double Deep Q-Network** (van Hasselt et al. 2016) extends the base DQN with a neural network for Q approximation and online/target separation to eliminate overestimation bias.

#### Base DQN Problem: overestimation bias

Classic DQN uses the **same network** to select and evaluate the optimal action:

$$\text{DQN: } y = r + \gamma \cdot \max_{a'} Q_{\theta^-}(s', a')$$

The `max` on the same estimation signal systematically amplifies errors → inflated Q estimates → suboptimal policy.

#### DDQN Solution: selection/evaluation separation

$$\text{DDQN: } y = r + \gamma \cdot Q_{\theta^-}\!\left(s',\; \underset{a'}{\arg\max}\; Q_\theta(s', a')\right)$$

- **Online network** $Q_\theta$ → selects which action is best
- **Target network** $Q_{\theta^-}$ → evaluates how much that action is worth

#### QNetwork Architecture

```
Input(33) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(30)

Total parameters: 22,174
Device: CPU (Colab free tier)
```

#### Replay Buffer

```python
# FIFO circular buffer with 50,000 transition capacity
# Obs are flattened on push: guaranteed shape (OBS_DIM_FLAT,)
buffer = deque(maxlen=50_000)

# Uniform random sampling → breaks temporal correlation
batch = random.sample(buffer, BATCH_SIZE)
```

#### DDQN Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `ALPHA_DDQN` | 0.0005 | Conservative LR for stability |
| `GAMMA` | 0.99 | Same as SARSA for direct comparison |
| `BATCH_SIZE` | 64 | Standard for MLP on small space |
| `BUFFER_SIZE` | 50,000 | Sufficient for 10k episodes |
| `TARGET_UPDATE` | 500 | Hard update every 500 steps |
| `HIDDEN_DIM` | 128 | Capacity/speed balance |

---

## Project Architecture

```
deepguard-rl/
│
├── DeepGuard_RL_CyberSecurity.ipynb   # Main notebook (Sections 0-3 + 4)
│
├── README.md                          # This file
│
├── results/
│   ├── sarsa_reward_curve.png         # SARSA reward on random_attack-v0
│   ├── sarsa_hack_rate.png            # SARSA hack rate
│   ├── ddqn_random_reward.png         # DDQN reward on random_attack-v0
│   ├── ddqn_random_hack_rate.png      # DDQN random hack rate
│   ├── ddqn_maximal_reward.png        # DDQN reward on maximal_attack-v0
│   ├── ddqn_maximal_hack_rate.png     # DDQN maximal hack rate
│   ├── sarsa_v3_reward_curve.png      # SARSA reward on random_attack-v3
│   ├── sarsa_v3_hack_rate.png         # SARSA v3 hack rate
│   ├── ddqn_v3_reward_curve.png       # DDQN reward on random_attack-v3
│   ├── ddqn_v3_hack_rate.png          # DDQN v3 hack rate
│   ├── comparison_hack_rate.png       # Hack rate comparison all scenarios
│   └── comparison_reward.png          # Reward comparison all scenarios
│
└── docs/
    └── architecture.png               # DDQN architecture diagram
```

### Notebook Structure

```
Section 0 — Setup & Configuration
    ├── 0-A  Dependency installation (with known fixes)
    ├── 0-A2 Installation verification
    ├── 0-B  Global imports and constants
    ├── 0-C  Environment exploration (detects actual OBS_DIM_FLAT)
    └── 0-D  Shared utility functions

Section 1 — SARSA
    ├── 1-A  Q-table, ε-greedy, sarsa_update
    ├── 1-B  Training loop (10k episodes on v0, 20k on v3)
    ├── 1-C  Results visualization
    └── 1-D  Quantitative analysis

Section 2 — DDQN
    ├── 2-A  Neural network architecture
    ├── 2-B  ReplayBuffer
    ├── 2-C  DDQNAgent (online + target net)
    ├── 2-D  Training on random_attack-v0
    ├── 2-E  Training on maximal_attack-v0
    └── 2-F  Results visualization

Section 3 — Comparative Analysis
    ├── 3-A  Visual comparison (hack rate + reward)
    ├── 3-B  Quantitative report
    └── 3-C  In-depth critical analysis

Section 4 — v0 vs v3 Comparison: Complexity Scaling
    ├── 4-A  v3 environment configuration and dimension detection
    ├── 4-B  SARSA training on random_attack-v3
    ├── 4-C  DDQN training on random_attack-v3
    ├── 4-D  v3 results visualization
    ├── 4-E  Direct v0 vs v3 comparison (hack rate + reward)
    └── 4-F  Comparative report and complexity effect analysis
```

---

## Results

### Note on hack_rate Calculation

`gym-idsgame` does not expose `info['attack_success']` in the dictionary returned by `step()`.
The hack_rate is calculated at episode level: an episode is considered compromised
if the defender receives at least one negative reward (`reward < 0`) during the episode.
The values reported below reflect this post-fix logic.

### Final Metrics — Sections 1-2 (greedy evaluation, 100 episodes every 500 training)

| Algorithm | Environment | Final hack rate | Final reward | Δ Reward |
|---|---|---|---|---|
| SARSA | random_attack-v0 | 0.000 | 0.660 | -0.020 |
| DDQN | random_attack-v0 | 0.000 | 0.600 | -0.060 |
| DDQN | maximal_attack-v0 | 0.000 | 0.760 | +0.060 |

### Final Metrics — Section 4 (v0 vs v3, greedy evaluation)

| Algorithm | Environment | Final hack rate | Final reward |
|---|---|---|---|
| SARSA | random_attack-v0 | 0.000 | 0.660 |
| DDQN  | random_attack-v0 | 0.000 | 0.600 |
| SARSA | random_attack-v3 | 0.000 | 0.780 |
| DDQN  | random_attack-v3 | 0.000 | 0.900 |

> The reported values are indicative of results observed in a single
> run of our experiments. In RL, different runs can produce different
> results. The effect of increased complexity on metrics must be verified
> empirically and cannot be assumed automatically.

### Reward Curves — SARSA (v0)

| Reward per episode | Hack rate over time |
|---|---|
| ![SARSA Reward v0](results/sarsa_reward_curve.png) | ![SARSA Hack Rate v0](results/sarsa_hack_rate.png) |

### Reward Curves — DDQN random_attack (v0)

| Reward per episode | Hack rate over time |
|---|---|
| ![DDQN Random Reward v0](results/ddqn_random_reward.png) | ![DDQN Random Hack Rate v0](results/ddqn_random_hack_rate.png) |

### Reward Curves — DDQN maximal_attack (v0)

| Reward per episode | Hack rate over time |
|---|---|
| ![DDQN Maximal Reward v0](results/ddqn_maximal_reward.png) | ![DDQN Maximal Hack Rate v0](results/ddqn_maximal_hack_rate.png) |

### Reward Curves — SARSA (v3)

| Reward per episode | Hack rate over time |
|---|---|
| ![SARSA Reward v3](results/sarsa_v3_reward_curve.png) | ![SARSA Hack Rate v3](results/sarsa_v3_hack_rate.png) |

### Reward Curves — DDQN random_attack (v3)

| Reward per episode | Hack rate over time |
|---|---|
| ![DDQN Reward v3](results/ddqn_v3_reward_curve.png) | ![DDQN Hack Rate v3](results/ddqn_v3_hack_rate.png) |

### Comparative Comparison

| Hack rate comparison | Reward comparison |
|---|---|
| ![Hack Rate Comparison](results/comparison_hack_rate.png) | ![Reward Comparison](results/comparison_reward.png) |

----
## Critical Analysis

### 1 — Hack_rate: Implementation Bug and Applied Fix

In the first version of the notebook, the hack_rate was artificially zero across all
scenarios due to an implementation bug: the code used `info.get('attack_success', False)`
but `gym-idsgame` does not expose this key — the `info` dictionary contains exclusively
`{'moved': bool}`.

After the fix, the hack_rate is calculated at episode level via an `episode_hacked` flag
set to `True` if the defender receives at least one negative reward during the episode.
The values observed in our experiments reflect this correct logic.

**Methodological implication:** hack_rate is a relevant metric for comparing
SARSA and DDQN, but its actual value depends on the environment configuration
and must be read from experimental results — it cannot be assumed a priori as a discriminator.

### 2 — DDQN-maximal Outperforms DDQN-random (Counterintuitive Result)

```
DDQN random_attack  → final reward 0.600, final loss ~1.55
DDQN maximal_attack → final reward 0.760, final loss ~0.003
```

**maximal_attack is deterministic**: it always attacks the same node with the same logic.
It is theoretically severe as an opponent, but its regularity makes it more predictable
and therefore easier for the defender to learn compared to a completely stochastic
attacker. This regularity allows the defender to converge on a
**stable and specific counter-strategy**, producing very low loss and high reward.

**random_attack introduces structural variance**: the defender must cover a
distribution of unpredictable attacks. The Bellman target varies more between batches
→ higher loss, slower convergence.

> In RL theory terms: maximal_attack produces an MDP with more stationary
> dynamics → easier to approximate with a neural network.

### 3 — DDQN-random Loss: Peak at ep 3500, Then Descent

```
ep  500  → loss  0.019  (buffer almost empty)
ep 3500  → loss  7.637  ← PEAK (maximum variance)
ep 9500  → loss  1.557  (convergence)
```

This is the **expected and healthy** behavior in DQN/DDQN:
- **Phase 1** (0-500): empty buffer, few updates, low loss
- **Phase 2** (500-3500): high ε, random policy, noisy targets → loss increases
- **Phase 3** (3500+): ε decreases, policy stabilizes, coherent signal → loss decreases

### 4 — SARSA: Dominant Action at 91.4% (v0) and 96.2% (v3)

```
v0 — Action  0: 16798 states (91.4%)
v3 — Action  0: ~55000 states (96.2%)
```

The Q-table has values almost all zero (std=0.0006 on v0). Action 0 is preferred
not because it is genuinely optimal, but because it is the first to receive minimal
positive reinforcement on rarely visited states — `argmax` returns index 0 in case of tie.
On v3 the degeneration is more pronounced, consistent with the intrinsic limit of tabular
approaches as state space complexity grows.

This confirms the **structural limit of tabular approaches**: lack of
generalization between similar states, degenerate policy on rarely visited states.

### 5 — Complexity Effect: v0 vs v3

DDQN is theoretically better suited to larger state spaces thanks to neural
generalization. The actual advantage over SARSA on v3 must be read from the metrics
observed in our experiments: increased complexity expands the observation and
action space, but it cannot be automatically assumed that it makes hack_rate more
discriminative or DDQN's advantage more pronounced.

## Key Insights

The main results observed in the experiments are as follows:

1. **hack_rate Metric Bug**
   - The `gym-idsgame` environment does not expose `attack_success` in the `info` dictionary.
   - Correct hack rate calculation requires an **episode-based** metric based on `reward < 0`.
   - Without this fix, the hack rate is artificially zero and leads to misleading conclusions.

2. **Deterministic vs Stochastic Attacker**
   - `maximal_attack` is more aggressive but also more regular and predictable.
   - This regularity allows the DDQN defender to learn a more stable counter-strategy compared to the `random_attack` case.

3. **Limits of Tabular Approaches**
   - SARSA works well as an interpretable baseline on small environments.
   - As state space complexity increases, the Q-table tends to become sparser and the policy more degenerate.

4. **Potential Advantage of Deep RL**
   - DDQN is theoretically better suited to larger observation spaces thanks to neural generalization.
   - The actual advantage over SARSA must however be verified empirically on run results.

## Lessons Learned

During project development, several practical lessons emerged:

- Academic RL environment APIs may differ from documentation or expected behavior.
- Before interpreting metrics, it is essential to verify they are aligned with the signals actually returned by the environment.
- In Deep RL, controlling the dimensions of observations, actions, and tensors is crucial to avoid architectural mismatches.
- Replay buffer, ε-greedy exploration, and random initialization introduce variability between runs.

This project shows that an important part of the work in Reinforcement Learning consists of validating environment, metrics, and experimental assumptions.

## Future Work

Possible project extensions:

- Testing on more complex environments (`v5`, `v10`);
- Running multiple runs with different seeds to estimate variance and robustness of results;
- Comparison with more advanced Deep RL architectures (Dueling DQN, PPO);
- Introduction of self-play scenarios or RL attackers to increase benchmark realism.

----

## Installation and Usage

### Requirements

- Google Colab (recommended) **or** local Python 3.12+
- GPU optional (the project runs correctly on CPU)

### Quick Start on Google Colab

1. Open the notebook on Google Colab via the badge at the top
2. Run the cells in order from Section 0
3. Wait for training to complete (~15-20 min on Colab CPU for Sections 1-3)

### Local Installation

```bash
# Clone this repo
git clone https://github.com/tuousername/deepguard-rl.git
cd deepguard-rl

# Clone gym-idsgame
git clone https://github.com/Limmen/gym-idsgame.git

# Install dependencies
pip install -e ./gym-idsgame --no-deps
pip install "gymnasium==0.26.3" numpy matplotlib seaborn tqdm
pip install torch torchvision opencv-python imageio
pip install jsonpickle tensorboard scikit-learn
pip install pyglet==1.5.15 --no-deps

# NumPy 2.0 fix (in every Python script before importing gym_idsgame)
# import numpy as np
# if not hasattr(np, 'bool8'): np.bool8 = np.bool_
```

### Execution

```bash
jupyter notebook DeepGuard_RL_CyberSecurity.ipynb
```

---

## Applied Technical Fixes

During development, the following issues were identified and resolved, documented for future reference:

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `gym_idsgame` not found after `pip install -e` | `pip install -e` does not update running Colab kernel's `sys.path` | `sys.path.insert(0, REPO_PATH)` before every import |
| 2 | `AttributeError: np.bool8` | Removed in NumPy 2.0 (June 2024) | `np.bool8 = np.bool_` monkey-patch before importing `gym_idsgame` |
| 3 | `TypeError: cannot unpack non-iterable int` | `step()` requires tuple `(att_action, def_action)` even in single-agent | Wrapper `env_step(env, defender_action)` with `(0, defender_action)` |
| 4 | `TypeError: unsupported operand type +=: float and tuple` | `step()` returns `reward` as tuple `(att_r, def_r)` | Extract `reward[1]` (defender) in `env_step` |
| 5 | `RuntimeError: mat1 shapes (64x33) cannot be multiplied (11x128)` | Actual `obs` has shape `(33,)` vs declared `(1,11)=11` | `OBS_DIM_FLAT = obs.flatten().shape[0]` detected from actual reset; `flat_obs()` applied everywhere |
| 6 | `hack_rate` always 0 across all scenarios | `info['attack_success']` not exposed by gym-idsgame (`info = {'moved': bool}`) | Episode-based calculation: flag `episode_hacked = True` if `reward < 0` during episode |
| 7 | Action out of range (e.g. 33) in SARSA v0 top action report | `zip(*np.unique(...))` ordered by action value (numeric index), not by frequency | `zip(counts, unique)` with `counts` first + `assert all(0 <= a < N_ACTIONS)` |

---

## References

| Resource | Link |
|---|---|
| `gym-idsgame` repository | [github.com/Limmen/gym-idsgame](https://github.com/Limmen/gym-idsgame) |
| Original paper (CNSM 2020) | [Finding Effective Security Strategies through RL and Self-Play](https://arxiv.org/abs/2009.08120) |
| Double DQN (van Hasselt et al. 2016) | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) |
| Original DQN (Mnih et al. 2015) | [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) |
| SARSA — Sutton & Barto | [Reinforcement Learning: An Introduction (ch. 6)](http://incompleteideas.net/book/the-book-2nd.html) |

---

## Author

**Francesco Scarano**  
Senior IT Manager | AI Engineering | Data & Digital Solutions

GitHub:  
https://github.com/Nimus74

LinkedIn:  
https://www.linkedin.com/in/francescoscarano/

---

## License

This project is licensed under the MIT License.
