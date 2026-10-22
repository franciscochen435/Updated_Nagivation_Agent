import sys
import time
import heapq
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from campus_gui import *
from campus_env import BASE_TRAVERSAL_TIMES

BOLD = "\033[1m"  # ANSI escape sequence for bold text
RESET = "\033[0m"  # ANSI escape sequence to reset text formatting

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv

env = game

def hash_obs(obs):
    px, py = obs['position']
    cx, cy = obs['nearest_crowd']

    # Nearest crowd relative to the agent, clipped to a 5x5 window
    if (cx, cy) == (-1, -1):# tunnel layer / no crowd
        crowd_rel = (-9, -9)
    else:
        dx, dy = cx - px, cy - py
        if abs(dx) <= 2 and abs(dy) <= 2:
            crowd_rel = (dx, dy)
        else:
            crowd_rel = (9, 9)# far away, treat as irrelevant

    return (px, py, obs['weather'], obs['layer'], crowd_rel)

def Q_learning(num_episodes=10000, gamma=0.99, epsilon=1, decay_rate=0.999, min_epsilon=0.05, min_eta=0.1, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    Q_table = {}
    num_updates = defaultdict(lambda: np.zeros(env.action_space.n))
    rewards = []
    crowd_ratios = []

    for episode in tqdm(range(num_episodes)):
        total_reward = 0
        obs, reward, done, info = env.reset()

        total_reward += reward

        # Track crowd for this episode
        episode_steps = 0
        crowd_steps = 0

        while not done:
            state = hash_obs(obs)

            # Check if agent is in crowd
            if obs.get('is_crowd', 0) == 1:
                crowd_steps += 1
            episode_steps += 1

            if state not in Q_table.keys():
                Q_table[state] = np.zeros(env.action_space.n)

            current_prob = random.random()
            if current_prob > epsilon:
                action = np.argmax(Q_table[state])
            else:
                action = random.randrange(env.action_space.n)  # seeded, unlike action_space.sample()

            # Floor the learning rate so later (more accurate) targets still matter
            eta = max(min_eta, 1 / (1 + num_updates[state][action]))

            obs, reward, done, info = env.step(action)
            total_reward +=reward
            state_next = hash_obs(obs)
            if state_next not in Q_table:
                Q_table[state_next] = np.zeros(env.action_space.n)

            # Values for Q_table update
            Q_old = Q_table[state][action]
            V_old = np.max(Q_table[state_next])

            # Update Q value for current (state, action, reward, next state) as well as number of updates
            Q_table[state][action] = (1 - eta) * Q_old + eta * (reward + (gamma * V_old))
            num_updates[state][action] += 1

        # Calculate crowd contact ratio for this episode
        if episode_steps > 0:
            crowd_ratio = crowd_steps / episode_steps
        else:
            crowd_ratio = 0
        crowd_ratios.append(crowd_ratio)

        # decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * decay_rate)
        rewards.append(total_reward)

    return Q_table, rewards, crowd_ratios


def moving_average(values, window=500):
    """Moving average tracks the current policy; a cumulative average lags behind early episodes."""
    window = min(window, len(values))
    return np.arange(window - 1, len(values)), np.convolve(values, np.ones(window) / window, mode='valid'), window


"""
Specify number of episodes, decay rate and number of seeds for training and evaluation.
Each seed trains an independent Q-table; evaluation reports mean ± std across seeds.
"""
num_episodes = 30000
decay_rate = 0.9998  # epsilon reaches ~0.05 around episode 15k
num_seeds = 3
num_eval_episodes = 10000


def q_table_filename(seed):
    return f"Q_table_{num_episodes}_{decay_rate}_seed{seed}.pickle"


"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-tables.
"""
if train_flag:
    all_rewards = []
    all_crowd_ratios = []
    for seed in range(num_seeds):
        print(f"Training seed {seed + 1}/{num_seeds}")
        Q_table, episode_rewards, crowd_ratios = Q_learning(
            num_episodes=num_episodes, gamma=0.99, epsilon=1, decay_rate=decay_rate, seed=seed
        )  # Run Q-learning

        # Save the Q-table dict to a file
        with open(q_table_filename(seed), "wb") as handle:
            pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
        all_rewards.append(episode_rewards)
        all_crowd_ratios.append(crowd_ratios)

    # Training reward curve, one moving average per seed
    plt.figure(figsize=(10, 6), dpi=300)
    for seed, episode_rewards in enumerate(all_rewards):
        x, avg, window = moving_average(episode_rewards)
        plt.plot(x, avg, linewidth=1.5, label=f'Seed {seed}')
    plt.title(f"Episode Total Reward (moving average over {window} episodes)", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_plot_{num_episodes}episodes_{decay_rate}_decay.png", dpi=300)
    print(f"Training reward plot saved as 'training_plot_{num_episodes}episodes_{decay_rate}_decay.png'")

    # Crowd contact ratio during training, one moving average per seed
    plt.figure(figsize=(10, 6), dpi=300)
    for seed, crowd_ratios in enumerate(all_crowd_ratios):
        x, avg, window = moving_average(crowd_ratios)
        plt.plot(x, avg, linewidth=1.5, label=f'Seed {seed}')
    plt.xlabel('Episode', fontsize=14, fontweight='bold')
    plt.ylabel('Crowd Contact Ratio', fontsize=14, fontweight='bold')
    plt.title(f'Crowd Contact Ratio During Training (moving average over {window} episodes)', fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig('crowd_contact_ratio_over_episodes.png', dpi=300, bbox_inches='tight')
    print("Crowd contact ratio plot saved as 'crowd_contact_ratio_over_episodes.png'")


"""
Shortest-path baseline: Dijkstra over (x, y, layer) using the same time costs as
CampusEnv.calculate_time_cost, assuming no crowds and no slips. The weather is known.
"""
MOVE_DELTAS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (-1, -1), 5: (1, -1), 6: (-1, 1), 7: (1, 1)}
TOGGLE_ACTION = 8
BUILDING_CODES = set(BUILDINGS.values())


def baseline_transitions(env, weather):
    """{(x, y, layer): [(action, next_node, time_cost), ...]} for every walkable cell."""
    maps = (env.surface_map, env.tunnel_map)
    transitions = {}
    for layer, grid in enumerate(maps):
        weather_mult = env.weather_multipliers["surface" if layer == 0 else "tunnel"][weather]
        for y in range(env.grid_height):
            for x in range(env.grid_width):
                cell = grid[y, x]
                if cell == WALL:
                    continue
                edges = []
                for action, (dx, dy) in MOVE_DELTAS.items():
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < env.grid_width and 0 <= ny < env.grid_height) or grid[ny, nx] == WALL:
                        continue
                    next_cell = grid[ny, nx]
                    cost = env.action_base_times[env.actions[action]] * BASE_TRAVERSAL_TIMES[next_cell]
                    if next_cell not in BUILDING_CODES:  # buildings are sheltered from weather
                        cost *= weather_mult
                    edges.append((action, (nx, ny, layer), cost))
                if cell in env.tunnel_building_codes:
                    edges.append((TOGGLE_ACTION, (x, y, 1 - layer), env.action_base_times["TOGGLE_LAYER"]))
                transitions[(x, y, layer)] = edges
    return transitions


def shortest_time_to_goal(env, transitions):
    """Reverse Dijkstra from every goal cell: minimum time from each node to the goal."""
    maps = (env.surface_map, env.tunnel_map)
    reverse = defaultdict(list)
    for node, edges in transitions.items():
        for _, next_node, cost in edges:
            reverse[next_node].append((node, cost))

    dist = {node: 0.0 for node in transitions if maps[node[2]][node[1], node[0]] == env.goal_building_code}
    heap = [(0.0, node) for node in dist]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for prev_node, cost in reverse[node]:
            if d + cost < dist.get(prev_node, float("inf")):
                dist[prev_node] = d + cost
                heapq.heappush(heap, (d + cost, prev_node))
    return dist


def baseline_action(transitions, dist, obs):
    node = (obs['position'][0], obs['position'][1], obs['layer'])
    return min(transitions[node], key=lambda e: e[2] + dist.get(e[1], float("inf")))[0]


"""
Evaluation mode: play episodes using the saved Q-tables and compare against the baseline.
Episode i is seeded with i for every policy, so all policies face the same start position,
layer and weather.
"""
WEATHERS = ["clear", "rain", "snow"]
WEATHER_LABELS = ["Clear", "Rain", "Snow"]
START_LAYERS = ["surface", "tunnel"]
EXPOSURE_TYPES = ["outdoor", "building", "tunnel"]


def cell_type(obs):
    """Where a step ended: outdoors on the surface (exposed to weather), inside a building, or in the tunnel."""
    if obs['layer'] == 1:
        return "tunnel"
    return "building" if obs['current_building'] > 0 else "outdoor"


def run_episode(policy, episode_seed, render=False):
    """Play one seeded episode with `policy(obs) -> action` and return its statistics."""
    random.seed(episode_seed)
    obs, _, done, info = env.reset()
    stats = {
        "weather": obs['weather'],
        "start_layer": START_LAYERS[obs['layer']],
        "reward": 0.0,
        "steps": 0,
        "layer_steps": {"surface": 0, "tunnel": 0},
        "toggle_opportunities": 0,
        "toggles": 0,
        "exposure": {t: 0 for t in EXPOSURE_TYPES},  # cells moved into, by type
        "crowd_contacts": 0,
    }
    while not done:
        stats["layer_steps"][START_LAYERS[obs['layer']]] += 1
        action = policy(obs)

        # Only count toggles from surface (layer 0) to tunnel (layer 1)
        if obs['can_toggle_layer'] == 1 and obs['layer'] == 0:
            stats["toggle_opportunities"] += 1
            stats["toggles"] += action == TOGGLE_ACTION

        prev_position = obs['position']
        obs, reward, done, info = env.step(action)
        stats["reward"] += reward
        stats["steps"] += 1
        if action in MOVE_DELTAS and obs['position'] != prev_position:
            stats["exposure"][cell_type(obs)] += 1
        if "Walked into a crowd" in info["result"]:
            stats["crowd_contacts"] += 1
        if render:
            refresh(obs, reward, done, info, delay=0.1)

    stats["success"] = not (info["timeout"] or info["truncated"])
    stats["time"] = obs['time']
    return stats


def make_agent_policy(Q_table, counters):
    def policy(obs):
        state = hash_obs(obs)
        if state in Q_table:
            counters["from_q_table"] += 1
            return int(np.argmax(Q_table[state]))  # Greedy action, same as exploitation in training
        counters["random"] += 1  # Fallback to random action if state not in Q-table
        counters["unseen_states"].add(state)
        return random.randrange(env.action_space.n)
    return policy


def summarize(agent_episodes, baseline_episodes):
    """Per-weather means for one seed: {weather: {metric: (agent, baseline)}}.
    Time is averaged over episodes where both reached the goal; everything else over all episodes."""
    summary = {}
    for weather in WEATHERS:
        pairs = [(a, b) for a, b in zip(agent_episodes, baseline_episodes) if a["weather"] == weather]
        both_succeeded = [(a, b) for a, b in pairs if a["success"] and b["success"]]

        def mean_pair(metric, subset=pairs):
            if not subset:
                return (float("nan"), float("nan"))
            return (np.mean([metric(a) for a, _ in subset]), np.mean([metric(b) for _, b in subset]))

        summary[weather] = {
            "success": mean_pair(lambda e: e["success"]),
            "time": mean_pair(lambda e: e["time"], both_succeeded),
            "reward": mean_pair(lambda e: e["reward"]),
            "crowd_contacts": mean_pair(lambda e: e["crowd_contacts"]),
            **{t: mean_pair(lambda e, t=t: e["exposure"][t]) for t in EXPOSURE_TYPES},
        }
    return summary


def across_seeds(summaries, metric, who):
    """{weather: (mean, std)} across seeds. who: 0 = agent, 1 = baseline."""
    return {
        weather: (np.mean([s[weather][metric][who] for s in summaries]), np.std([s[weather][metric][who] for s in summaries]))
        for weather in WEATHERS
    }


def grouped_bar_chart(agent_stats, baseline_stats, ylabel, title, filename, value_format='{:.2f}'):
    """Agent (mean ± std across seeds) next to the baseline, one group per weather."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(WEATHERS))
    width = 0.38
    series = [
        (-width / 2, agent_stats, f'Q-learning agent (mean ± std, {num_seeds} seeds)', 'tab:blue'),
        (width / 2, baseline_stats, 'Shortest-path baseline', 'lightgray'),
    ]
    for offset, stats, label, color in series:
        means = [stats[w][0] for w in WEATHERS]
        stds = [stats[w][1] for w in WEATHERS]
        ax.bar(x + offset, means, width, yerr=stds, capsize=6, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        for i in range(len(WEATHERS)):
            ax.text(x[i] + offset, means[i] + stds[i], value_format.format(means[i]), ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_LABELS, fontsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    top = max(m + s for stats in (agent_stats, baseline_stats) for m, s in stats.values())
    ax.set_ylim([0, top * 1.25 if top > 0 else 1])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")


if not train_flag:
    filenames = [q_table_filename(seed) for seed in range(num_seeds)]
    input(
        f"\n{BOLD}Currently loading Q-tables from "
        + ", ".join(filenames)
        + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes, decay_rate and num_seeds in Q_learning.py)."
    )

    baseline_plans = {}
    for weather in WEATHERS:
        transitions = baseline_transitions(env, weather)
        baseline_plans[weather] = (transitions, shortest_time_to_goal(env, transitions))

    def baseline_policy(obs):
        transitions, dist = baseline_plans[obs['weather']]
        return baseline_action(transitions, dist, obs)

    # The baseline is deterministic given the episode seed, so one run serves every agent seed
    print("Evaluating shortest-path baseline")
    baseline_episodes = [run_episode(baseline_policy, episode) for episode in tqdm(range(num_eval_episodes))]

    agent_runs = []
    for seed, filename in enumerate(filenames):
        with open(filename, "rb") as handle:
            Q_table = pickle.load(handle)
        counters = {"from_q_table": 0, "random": 0, "unseen_states": set()}
        policy = make_agent_policy(Q_table, counters)

        print(f"\n{BOLD}Evaluating {filename}{RESET}")
        start_time = time.perf_counter()
        episodes = [run_episode(policy, episode, render=gui_flag and seed == 0) for episode in tqdm(range(num_eval_episodes))]
        elapsed_time = time.perf_counter() - start_time
        agent_runs.append(episodes)

        total_actions = counters["from_q_table"] + counters["random"]
        print(f"Number of unique states in Q-table: {len(Q_table)}")
        print(f"Average reward over {num_eval_episodes} episodes: {np.mean([e['reward'] for e in episodes]):.2f}")
        print(f"Average episode length: {np.mean([e['steps'] for e in episodes]):.2f}")
        print(f"Evaluation time: {elapsed_time:.2f} seconds")
        print(f"Number of unique states reached in evaluation not seen in training: {len(counters['unseen_states'])}")
        print(f"Share of actions taken from Q-table: {counters['from_q_table'] / total_actions:.4%}")

    summaries = [summarize(episodes, baseline_episodes) for episodes in agent_runs]

    def stat(metric, who):
        return across_seeds(summaries, metric, who)

    # Agent vs shortest-path baseline, agent as mean ± std across seeds
    def fmt(mean_std, value_format):
        mean, std = mean_std
        return f"{value_format.format(mean)} ± {value_format.format(std)}"

    print(f"\n{BOLD}Agent (mean ± std over {num_seeds} seeds) vs shortest-path baseline{RESET}")
    rows = [
        ("Success rate", "success", "{:.2%}"),
        ("Time (both reached goal)", "time", "{:.2f}"),
        ("Total reward", "reward", "{:.2f}"),
        ("Crowd contacts / episode", "crowd_contacts", "{:.3f}"),
        ("Outdoor steps / episode", "outdoor", "{:.2f}"),
        ("Building steps / episode", "building", "{:.2f}"),
        ("Tunnel steps / episode", "tunnel", "{:.2f}"),
    ]
    for weather in WEATHERS:
        episodes_count = sum(e["weather"] == weather for e in baseline_episodes)
        print(f"\n{weather.capitalize()} ({episodes_count} episodes)")
        print(f"  {'Metric':<26} {'Agent':>22} {'Baseline':>12}")
        for label, metric, value_format in rows:
            agent_value = fmt(stat(metric, 0)[weather], value_format)
            baseline_value = value_format.format(stat(metric, 1)[weather][0])
            print(f"  {label:<26} {agent_value:>22} {baseline_value:>12}")

    # Visualize data to analysis agent performance
    # Layer usage and toggle charts pool the agent's episodes from all seeds
    all_agent_episodes = [e for episodes in agent_runs for e in episodes]

    # Generate layer usage visualization
    # Split by starting layer: half the episodes start in the tunnel, which inflates tunnel usage on its own
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    x = np.arange(3)
    width = 0.6

    for ax, start in zip(axes, START_LAYERS):
        tunnel_values = []
        surface_values = []
        for weather in WEATHERS:
            selected = [e for e in all_agent_episodes if e["start_layer"] == start and e["weather"] == weather]
            surface_count = sum(e["layer_steps"]["surface"] for e in selected)
            tunnel_count = sum(e["layer_steps"]["tunnel"] for e in selected)
            total = surface_count + tunnel_count
            if total > 0:
                tunnel_ratio = tunnel_count / total
                surface_ratio = surface_count / total
            else:
                tunnel_ratio = 0
                surface_ratio = 0
            tunnel_values.append(tunnel_ratio)
            surface_values.append(surface_ratio)

        ax.bar(x, tunnel_values, width, label='Tunnel', color='red',
        alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.bar(x, surface_values, width, bottom=tunnel_values, label='Surface', color='blue',
        alpha=0.8, edgecolor='black', linewidth=1.5)

        for i in range(3):
            t_val = tunnel_values[i]
            s_val = surface_values[i]
            ax.text(i, t_val / 2, f'{t_val:.2%}', ha='center', va='center', fontweight='bold', fontsize=11, color='white')
            ax.text(i, t_val + s_val / 2, f'{s_val:.2%}', ha='center', va='center', fontweight='bold', fontsize=11, color='black')

        ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
        ax.set_title('Episodes Starting on the Surface' if start == 'surface' else 'Episodes Starting in the Tunnel', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(WEATHER_LABELS, fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Usage Rate (share of steps)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12, framealpha=0.9)  # outside, bars fill the axes
    fig.suptitle('Agent Surface vs Tunnel Usage by Starting Layer and Weather', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('layer_usage_comparison.png', dpi=300, bbox_inches='tight')
    print("Layer usage comparison chart saved as 'layer_usage_comparison.png'")

    # Generate toggle probability visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.6

    toggle_probabilities = []
    for weather in WEATHERS:
        selected = [e for e in all_agent_episodes if e["weather"] == weather]
        opportunities = sum(e["toggle_opportunities"] for e in selected)
        toggle_choices = sum(e["toggles"] for e in selected)
        if opportunities > 0:
            probability = toggle_choices / opportunities
        else:
            probability = 0
        toggle_probabilities.append(probability)

    colors = ['lightgreen', 'skyblue', 'lightgray']

    bars = ax.bar(x, toggle_probabilities, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for i in range(3):
        val = toggle_probabilities[i]
        ax.text(i, val * 1.05, f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Toggle Probability', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title('Agent TOGGLE_LAYER Probability at Toggle Buildings Across Weather Conditions', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_LABELS, fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    max_prob = max(toggle_probabilities) if max(toggle_probabilities) > 0 else 0.01
    ax.set_ylim([0, max_prob * 1.2])

    plt.tight_layout()
    plt.savefig('toggle_probability_comparison.png', dpi=300, bbox_inches='tight')
    print("Toggle probability comparison chart saved as 'toggle_probability_comparison.png'")

    # Generate agent vs baseline comparisons
    grouped_bar_chart(
        stat("time", 0), stat("time", 1),
        'Average Time', 'Average Completion Time: Agent vs Shortest-Path Baseline',
        'average_time_comparison.png',
    )

    # Both rewards sit near the goal reward, so plot the penalty actually incurred
    # (goal reward - total reward): time cost plus weather/crowd/building penalties
    goal_reward = env.rewards["goal"]

    def penalty(stats):
        return {w: (goal_reward - mean, std) for w, (mean, std) in stats.items()}

    grouped_bar_chart(
        penalty(stat("reward", 0)), penalty(stat("reward", 1)),
        f'Average Penalty ({goal_reward} − total reward)', 'Average Penalty per Episode (lower is better)',
        'reward_comparison.png',
    )

    grouped_bar_chart(
        stat("crowd_contacts", 0), stat("crowd_contacts", 1),
        'Crowd Contacts per Episode', 'Crowd Contacts per Episode (lower is better)',
        'crowd_contact_comparison.png', value_format='{:.3f}',
    )

    # Generate weather exposure visualization: where each step ended, agent and baseline side by side
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(WEATHERS))
    width = 0.38
    exposure_colors = {"outdoor": "tab:orange", "building": "tab:green", "tunnel": "tab:red"}
    exposure_labels = {"outdoor": "Outdoors (exposed)", "building": "Inside buildings", "tunnel": "Tunnel"}

    for offset, who, name in [(-width / 2, 0, "Agent"), (width / 2, 1, "Baseline")]:
        bottom = np.zeros(len(WEATHERS))
        for exposure in EXPOSURE_TYPES:
            stats = stat(exposure, who)
            means = np.array([stats[w][0] for w in WEATHERS])
            ax.bar(x + offset, means, width, bottom=bottom, color=exposure_colors[exposure], alpha=0.85 if who == 0 else 0.45,
                   edgecolor='black', linewidth=1.2, label=exposure_labels[exposure] if who == 0 else None)
            for i in range(len(WEATHERS)):
                if means[i] >= 0.4:
                    ax.text(x[i] + offset, bottom[i] + means[i] / 2, f'{means[i]:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')
            bottom += means
        for i in range(len(WEATHERS)):
            ax.text(x[i] + offset, bottom[i], name, ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Steps per Episode', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weather Condition', fontsize=14, fontweight='bold')
    ax.set_title(f'Where Each Step Ends: Agent (solid, mean of {num_seeds} seeds) vs Baseline (faded)', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(WEATHER_LABELS, fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)

    plt.tight_layout()
    plt.savefig('weather_exposure_comparison.png', dpi=300, bbox_inches='tight')
    print("Weather exposure comparison chart saved as 'weather_exposure_comparison.png'")
