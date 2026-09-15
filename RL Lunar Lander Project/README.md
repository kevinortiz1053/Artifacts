# Beyond the Launchpad: Curriculum Learning in Lunar Lander

A comparison of Q-learning, SARSA, and n-step SARSA — with and without curriculum learning — on the OpenAI Gymnasium `LunarLander-v2` environment.

This repo contains the code and final report for our CS 138 final project, where we investigated whether training an agent on a curriculum of increasingly difficult environments (varying wind and turbulence) helps it land faster and more reliably than training directly on the full-difficulty environment.

## Overview

`LunarLander-v2` tasks an agent with safely landing a spacecraft using four discrete actions (do nothing, fire left engine, fire main engine, fire right engine). Rewards are sparse — the agent only gets strong signal on a successful landing (+100) or a crash (-100) — which makes it a good testbed for curriculum learning.

We built curricula that gradually ramp up wind power and turbulence power over the course of training, then compared the resulting policies against baselines trained directly on the full-difficulty environment (wind power = 15, turbulence power = 2).

**Hypothesis:** Curriculum-trained agents will learn faster and land more often than baseline agents, given the same total number of training episodes.

**Result:** Supported. Across all three algorithms, curriculum learning improved the number of successful landings within a fixed training budget. SARSA (1-step) was the strongest performer overall.

| Method | Landings (10k training episodes) | Landings (5k eval episodes, frozen policy) |
|---|---|---|
| Q-learning | 6 | 8 |
| Q-learning + Curriculum | 10 | 7 |
| SARSA | 60 | 70 |
| SARSA + Curriculum | 350 | 75 |
| N-step SARSA | 40 | 50 |
| N-step SARSA + Curriculum | 360 | 60 |

Full methodology, plots, and discussion are in the write-up: **`Final_Project_CS_138.pdf`**.

## Repository Contents

| File | Description |
|---|---|
| `Final_Project_CS_138.pdf` | Full project report — background, methodology, curricula design, results, and discussion. |
| `q-learning_final.py` | Baseline and curriculum Q-learning implementation, including hyperparameter (alpha) sweeps and averaged multi-run plots. |
| `sarsa_final.py` | Baseline (non-curriculum) 1-step SARSA implementation and hyperparameter sweeps. |
| `curriculum_sarsa_final.py` | Curriculum version of 1-step SARSA, split into five phases (`sarsa_c1` through `sarsa_c4`, then the full-difficulty phase) with increasing wind/turbulence. |
| `n-step_sarsa_final.py` | Baseline n-step SARSA implementation, including n-step return computation and hyperparameter sweeps. |
| `curriculum_n-sarsa_final.py` | Curriculum version of n-step SARSA. |
| `5krun_animations.py` | Loads a saved (pickled) policy and evaluates it over 5,000 frozen-policy episodes; also supports rendering a live animation of the lander using a trained policy. |

## Method Summary

- **State representation:** The environment returns an 8-tuple `[x, y, x_vel, y_vel, angle, angular_vel, left_leg_down, right_leg_down]`. To keep the state space tractable for tabular methods, each state is discretized: positions and angular velocity are rounded to the nearest whole number, velocities and angle are rounded to one decimal place, and the two boolean leg-contact flags are left as-is.
- **Policy:** Epsilon-greedy (ε = 0.1) over a Q-table implemented as a dictionary keyed by the stringified discretized state.
- **Curricula:** Three curricula were designed, each moving from `wind_power=0, turbulence_power=0` up to the baseline `wind_power=15, turbulence_power=2` over five phases. Curriculum 3 (which increases wind before turbulence, in smaller increments) performed best and is the one used in the final scripts.
- **Evaluation:** Each algorithm was trained for 10,000 episodes, repeated over 10 runs, and the landing counts were averaged. Learned policies were then frozen and evaluated on an additional 5,000 episodes to measure "true" policy quality independent of exploration-driven landings during training.

## Requirements

```bash
pip install gymnasium[box2d] matplotlib
```

`LunarLander-v2` requires the Box2D physics engine, installed via the `box2d` extra above.

## Usage

Each script is self-contained and runs its experiment (including plotting) when executed directly:

```bash
python sarsa_final.py
python curriculum_sarsa_final.py
python n-step_sarsa_final.py
python curriculum_n-sarsa_final.py
python q-learning_final.py
```

Training scripts that save policies (e.g. `sarsa_final.py`) will pickle the learned Q-table (e.g. `sarsa_baseline.pkl`) to the working directory. `5krun_animations.py` expects a pickled policy file (`curr_sarsa_baseline.pkl` by default) to be present, and will run 5,000 frozen-policy evaluation episodes, plotting landing counts over time. Uncomment the `for_animation(...)` call in that file to instead watch the lander fly with rendering enabled.

## Notes / Limitations

- Runs use `env.reset(seed=123)` for reproducibility across comparisons; some baseline scripts run unseeded for contrast (see the report for the seeded vs. unseeded comparison).
- The state discretization scheme was tuned by trial and error and is not guaranteed to be optimal — see the report's "Future Work" section for ideas on improving it (e.g. reward shaping based on pre-scaled state, alternative discretization granularity).
- Training is CPU-bound and tabular, so runtimes for 10,000-episode runs (×10 for averaging) can be significant, especially for n-step SARSA and the curriculum variants.

## References

See `Final_Project_CS_138.pdf` for full citations, including Bengio et al.'s original curriculum learning paper and Narvekar et al.'s survey on curriculum learning for RL.

## Authors

Kevin Ortiz, Brandon Wilson — CS 138 Final Project, May 2023
