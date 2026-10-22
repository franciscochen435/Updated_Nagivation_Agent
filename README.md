# Updated_Nagivation_Agent

A tabular Q-learning agent that navigates a 20×20 grid map of the Northeastern University campus. The map has two layers: the **surface** and the **underground tunnels**. The agent must reach a goal building (Snell Library by default) as quickly as possible under varying weather and crowd conditions, learning to take the tunnels in rain or snow and to avoid crowds.

## Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training and Evaluation](#training-and-evaluation)
- [Environment](#environment)
- [Main Changes](#main-changes)

## Project Structure

```
.
├── campus_env.py      # Gym environment: map, actions, time cost, reward function
├── campus_gui.py      # pygame visualization
├── Q_learning.py      # Q-learning training and evaluation script
└── requirements.txt
```

## Setup

Dependencies are pinned in `requirements.txt`. `numpy==1.24.3` only supports **Python 3.8–3.11**; it will not install on Python 3.12 or later.

```bash
git clone https://github.com/franciscochen435/Updated_Nagivation_Agent.git
cd Updated_Nagivation_Agent

# Create a virtual environment (Python 3.11 shown here)
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Training and Evaluation

```bash
# Train: writes Q_table_<num_episodes>_<decay_rate>.pickle and training plots
python Q_learning.py train

# Evaluate: loads the Q-table above, runs 10,000 episodes and writes summary plots
python Q_learning.py

# Evaluate with a pygame window to watch the agent
python Q_learning.py gui
```

`num_episodes` and `decay_rate` are set in `Q_learning.py`. Evaluation builds the Q-table filename from these two values, so training and evaluation must use the same values.

### Output Files

| File | Produced by | Contents |
|---|---|---|
| `Q_table_*.pickle` | Training | The trained Q-table |
| `training_plot_*.png` | Training | Total reward per episode and its running average |
| `crowd_contact_ratio_over_episodes.png` | Training | Fraction of steps spent in a crowd, per episode |
| `layer_usage_comparison.png` | Evaluation | Surface vs. tunnel usage under each weather condition |
| `toggle_probability_comparison.png` | Evaluation | Probability of entering the tunnel when standing in a toggle building, per weather condition |
| `average_time_comparison.png` | Evaluation | Average episode time per weather condition |

## Environment

### Actions (10)

Eight movement directions (orthogonal moves cost 1.0, diagonal moves cost 1.414), `TOGGLE_LAYER` (switch between surface and tunnel inside a building with a tunnel entrance), and `WAIT`. Each move has a 1% chance of slipping into a random adjacent cell.

### Time Cost

Time consumed per step = action base time × cell type factor × weather factor × crowd factor.

- **Cell type:** open ground 1.0, tunnel 1.5, building 2.0.
- **Weather (outdoor surface only):** clear 1.0, rain 1.3, snow 1.6.
- **Crowd:** low 1.0, medium 1.2, high 1.5.

An episode is cut off when time reaches 400 or the step count reaches 1000.

### Reward Function

| Term | Value | When it applies |
|---|---|---|
| `goal` | +1000 | Reaching the goal building |
| `time_cost` | −2 per time unit | Every step, based on the time actually consumed (moves, `TOGGLE_LAYER` and `WAIT`) |
| `crowd` | −10 | Stepping into a crowded cell on the surface |
| `rain_penalty` / `snow_penalty` | −4 / −5 | Moving outdoors on the surface in rain / snow |
| `bad_movement` | −5 | Moving in the tunnel when the weather is clear |
| `enter_wrong_building` | −3 | Entering a non-goal building on the surface |
| `invalid_action` | −5 | Hitting a wall, going out of bounds, or `TOGGLE_LAYER` where no toggle is possible |
| Layer switch | −50 | Leaving the tunnel for the surface in bad weather, or entering the tunnel in clear weather |
| `timeout` | 0 | Time limit or step limit reached (delay is already penalized through `time_cost`) |

### State Representation

```python
(x, y, weather, layer, crowd_rel)
```

`crowd_rel` is the offset `(dx, dy)` of the nearest crowd relative to the agent, kept only inside a 5×5 window. Crowds outside the window are encoded as `(9, 9)`; on the tunnel layer (where there are no crowds) it is `(-9, -9)`.

## Main Changes

### Reward Function (`campus_env.py`)

- **Removed the +10 reward for every tunnel step.** In bad weather the agent could farm reward by walking back and forth in the tunnel, earning more than it would by reaching the goal, so it learned to loop instead of navigating. The preference for tunnels in bad weather now comes only from the weather penalty applied on the surface.
- **`move_player` now accumulates reward terms instead of returning early.** Previously it returned on the first matching condition, so the crowd penalty and the wrong-building penalty were skipped in some cases. All applicable terms are now summed, and `info["result"]` joins multiple messages with `; `.
- **Time reward is now charged per step.** Previously a one-off `(max_time - time) × 2` bonus was paid only on reaching the goal. Because time is not part of the state, that signal was very noisy for the agent, and episodes that never reached the goal learned nothing about being fast. Each step now costs `−2 × time consumed`, and the goal pays a fixed +1000. For episodes that reach the goal the return differs only by a constant 800, so the preference ordering is unchanged. This also fixes `WAIT` being free, which made standing still look attractive.
- **Toggle reward reduced from ±300/350 to 0.** Those values were two orders of magnitude larger than the per-step rewards, so the agent was effectively optimizing for entering the tunnel rather than for reaching the goal. Only the −50 for switching in the wrong direction remains.
- **Timeout penalty changed from −800 to 0.** Delay is already penalized by the per-step time cost (roughly −800 accumulated by the time limit), so the extra −800 double-counted it. And since the state has no time information, the penalty landed on whatever cell the agent happened to be in at the cutoff, corrupting that cell's Q-values.

### Q-learning (`Q_learning.py`)

- **State representation:** the nearest crowd is now encoded as a relative offset clipped to a 5×5 window instead of absolute coordinates, and the redundant `can_toggle_layer` was removed. Previously a distant crowd moving by one cell produced a brand-new state, pushing the state count into the hundreds of thousands; the crowd component now has at most 27 values.
- **Hyperparameters:**

  | Parameter | Before | After | Reason |
  |---|---|---|---|
  | `gamma` | 0.9 | 0.99 | 0.9^25 ≈ 0.07, so the goal reward barely reached distant start positions |
  | `decay_rate` | 0.99999 | 0.99996 | ε was still about 0.37 at the end of training; it now reaches 0.05 around episode 75k |
  | `min_epsilon` | none | 0.05 | Keeps a small amount of exploration |
  | Learning rate | `1/(1+n)` | `max(0.1, 1/(1+n))` | `1/(1+n)` decays too fast; Q-learning targets keep changing, so early bad estimates were hard to correct |

> Because the state format has changed, existing `Q_table_*.pickle` files are no longer compatible. Retrain before evaluating.
