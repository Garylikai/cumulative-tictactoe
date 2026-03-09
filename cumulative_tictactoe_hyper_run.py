#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang (zhangshangtong.cpp@gmail.com)          #
# 2016 Jan Hakenberg (jan.hakenberg@gmail.com)                        #
# 2016 Tian Jun (tianjun.cpp@gmail.com)                               #
# 2016 Kenta Shimada (hyperkentakun@gmail.com)                        #
# 2022 - 2026 Kai Li (kai.li@stonybrook.edu)                          #
# 2026 Wei Zhu (wei.zhu@stonybrook.edu)                               #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# -----------------------------------------------------------------------
# cumulative_tictactoe_hyper_run.py
#
# Hyperparameter training / analysis script for cumulative tic-tac-toe.
# Imports core game implementation from cumulative_tictactoe_core.py.
# -----------------------------------------------------------------------

import time
import json
import csv
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind

# Import core classes and functions (must be in the same directory)
from cumulative_tictactoe_core import Player, Judger

# Training function
def train(episodes, window_size=20000, perf_threshold=1e-8, seed=None, epsilon=0.1, step_size=0.9, heuristic="zero"):
    # Train two TD agents via self-play.
    np.random.seed(seed)
    
    player1 = Player(epsilon, step_size, heuristic)
    player2 = Player(epsilon, step_size, heuristic)
    judger = Judger(player1, player2)
    
    # Counters for performance metrics
    p1_wins = 0
    p2_wins = 0
    draws = 0
    # Record running performance after each episode as (win_rate, draw_rate)
    performance_history = []
    episodes_to_converge = None

    for episode in range(1, episodes + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1
        
        # Calculate running averages
        p1_win_rate = p1_wins / episode
        draw_rate = draws / episode
        performance_history.append((p1_win_rate, draw_rate))
               
        player1.backup()
        player2.backup()
        judger.reset()

        # Check for convergence if we have enough episodes
        if episode >= window_size:
            # Consider the last window_size episodes
            recent_history = performance_history[-window_size:]
            # Separate win rates and draw rates
            win_rates_window = np.array([win[0] for win in recent_history])
            draw_rates_window = np.array([draw[1] for draw in recent_history])
            
            # Compute sample variance over the window. A low sample variance suggests stabilization.
            win_var = np.var(win_rates_window, ddof=1)
            draw_var = np.var(draw_rates_window, ddof=1)
            
            if win_var < perf_threshold and draw_var < perf_threshold:
                episodes_to_converge = episode
                print("Convergence achieved at episode %d with random seed %d and heuristic %s" %
                      (episodes_to_converge, seed, heuristic))
                break

    if episodes_to_converge is None:
        print("Convergence not reached in %d episodes." % episodes)
    return episodes_to_converge

def _save_results_csv(results, csv_path):
    # Save list-of-dicts to CSV (header derived from keys).
    if not results:
        return
    keys = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def _save_results_json(results, json_path):
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    # Experiment configuration
    # Grid-search parameters (adjust as needed)
    SEEDS = list(range(121, 126))
    EPSILONS = [0.05, 0.10, 0.20, 0.30]
    STEP_SIZES = [0.05, 0.10, 0.20, 0.50]
    HEURISTIC_FOR_GRID = "zero"   # choose "zero", "tcd", or "random" as needed

    # How many episodes to run per grid configuration.
    EPISODES_GRID = int(1e6)

    # Table A1: Run grid search and log results
    # grid_results is a list of per-run dicts (seed, epsilon, step_size, heuristic, episodes_to_converge, ...)
    grid_results = []
    total_runs = len(SEEDS) * len(EPSILONS) * len(STEP_SIZES)
    run_counter = 0
    print(f"Starting grid search: {total_runs} runs (episodes per run = {EPISODES_GRID})")

    for seed in SEEDS:
        for eps in EPSILONS:
            for alpha in STEP_SIZES:
                run_counter += 1
                print(f"\nRun {run_counter}/{total_runs}: seed={seed}, epsilon={eps}, alpha={alpha}, heuristic={HEURISTIC_FOR_GRID}")
                start_time = time.time()
                episodes_to_converge = train(EPISODES_GRID, seed=seed, epsilon=eps, step_size=alpha, heuristic=HEURISTIC_FOR_GRID)
                elapsed = time.time() - start_time

                grid_results.append({
                    "seed": seed,
                    "epsilon": eps,
                    "step_size": alpha,
                    "heuristic": HEURISTIC_FOR_GRID,
                    "episodes_requested": EPISODES_GRID,
                    "episodes_to_converge": episodes_to_converge,
                    "elapsed_seconds": round(elapsed, 2)
                })

                # Flush partial results to disk after each run to avoid data loss
                _save_results_csv(grid_results, "hyper_grid_results.csv")
                _save_results_json(grid_results, "hyper_grid_results.json")

    print(f"\nGrid search completed. Results saved to hyper_grid_results.csv and hyper_grid_results.json")

    # Table 3: mean ± SD per (epsilon, alpha) under HEURISTIC_FOR_GRID
    zero_summary = []
    for eps in EPSILONS:
        for alpha in STEP_SIZES:
            vals = []
            for rec in grid_results:
                if (rec.get("epsilon") == eps and
                    rec.get("step_size") == alpha and
                    rec.get("heuristic") == HEURISTIC_FOR_GRID):
                    v = rec.get("episodes_to_converge")
                    # Treat None (no convergence) as the cap for reporting; to exclude instead, skip these:
                    if v is None:
                        v = EPISODES_GRID
                    vals.append(float(v))

            if len(vals) == 0:
                mean_int = None
                sd_int = None
                n_runs = 0
            else:
                arr = np.array(vals, dtype=float)
                mean_val = float(np.mean(arr))
                # sample SD (ddof=1) if at least 2 samples, otherwise 0
                sd_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
                mean_int = int(round(mean_val))
                sd_int = int(round(sd_val))
                n_runs = arr.size

            zero_summary.append({
                "epsilon": eps,
                "step_size": alpha,
                "mean": mean_int,
                "sd": sd_int,
                "n_runs": n_runs
            })

    # Save table3 to CSV
    with open("zero_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epsilon","step_size","mean","sd","n_runs"])
        writer.writeheader()
        for r in zero_summary:
            writer.writerow(r)

    print("\nTable 3 (Mean ± SD, rounded)")
    for eps in EPSILONS:
        cells = []
        for alpha in STEP_SIZES:
            entry = next((x for x in zero_summary if x["epsilon"] == eps and x["step_size"] == alpha), None)
            if entry is None or entry["mean"] is None:
                cells.append("N/A")
            else:
                cells.append(f"{entry['mean']:,} ($\\pm$ {entry['sd']:,})")
        print(f"epsilon={eps}: " + "  &  ".join(cells))

    # Heuristic comparison
    SEEDS_HEUR_COMPARE = list(range(126, 131))
    HEURISTICS_TO_COMPARE = ["random", "tcd"]

    EPISODES_HEUR_COMPARE = int(1e6)
    
    # Table A2: Heuristic results
    # heuristics_results is a list of per-run dicts (seed, epsilon, step_size, heuristic, episodes_to_converge, ...)
    heuristics_results = []
    total_runs = len(SEEDS_HEUR_COMPARE) * len(HEURISTICS_TO_COMPARE)
    run_counter = 0
    print(f"\nStarting heuristic comparison: {total_runs} runs (episodes per run = {EPISODES_HEUR_COMPARE})")

    # heur_results is a dict with lists of episodes_to_converge per heuristic
    heur_results = {h: [] for h in HEURISTICS_TO_COMPARE}
    for seed in SEEDS_HEUR_COMPARE:
        for heur in HEURISTICS_TO_COMPARE:
            run_counter += 1
            print(f"\nRun {run_counter}/{total_runs}: seed={seed}, epsilon={0.05}, alpha={0.5}, heuristic={heur}")
            start = time.time()
            episodes_to_converge = train(EPISODES_HEUR_COMPARE, seed=seed, epsilon=0.05, step_size=0.5, heuristic=heur)
            elapsed = time.time() - start

            heuristics_results.append({
                "seed": seed,
                "epsilon": 0.05,
                "step_size": 0.5,
                "heuristic": heur,
                "episodes_requested": EPISODES_HEUR_COMPARE,
                "episodes_to_converge": episodes_to_converge,
                "elapsed_seconds": round(elapsed, 2)
            })

            # Flush partial results to disk after each run to avoid data loss
            _save_results_csv(heuristics_results, "heuristic_compare_results.csv")
            _save_results_json(heuristics_results, "heuristic_compare_results.json")

            # record numeric episodes_to_converge (if None, record EPISODES_HEUR_COMPARE)
            heur_results[heur].append(episodes_to_converge if episodes_to_converge is not None else EPISODES_HEUR_COMPARE)

    print(f"\nHeuristic comparison search completed. Results saved to heuristic_compare_results.csv and heuristic_compare_results.json")

    # If we have data for both heuristics, perform statistical tests
    a = np.array(heur_results[HEURISTICS_TO_COMPARE[0]])
    b = np.array(heur_results[HEURISTICS_TO_COMPARE[1]])
    print("\nStatistical tests on episodes_to_converge:")
    try:
        print(f"Samples for {HEURISTICS_TO_COMPARE[0]}: {list(a)[:5]}")
        print(f"Samples for {HEURISTICS_TO_COMPARE[1]}: {list(b)[:5]}")

        shapiro_p_a = shapiro(a).pvalue
        shapiro_p_b = shapiro(b).pvalue
        levene_p = levene(a, b).pvalue
        ttest_p = ttest_ind(a, b, alternative="greater").pvalue

        print(f"Shapiro p-value for {HEURISTICS_TO_COMPARE[0]}: {shapiro_p_a:.4f}, for {HEURISTICS_TO_COMPARE[1]}: {shapiro_p_b:.4f}")
        print(f"Levene p-value (equal variance): {levene_p:.4f}")
        print(f"One-sided t-test ({HEURISTICS_TO_COMPARE[0]} > {HEURISTICS_TO_COMPARE[1]}) p-value: {ttest_p:.4f}")
    except Exception as e:
        print("Could not complete statistical tests (insufficient or invalid data):", str(e))

    # Table 4: compare Random vs TCD at eps=0.05, alpha=0.5
    if len(heuristics_results) > 0 and all(h in heur_results for h in HEURISTICS_TO_COMPARE):
        baseline = HEURISTICS_TO_COMPARE[0]  # expected "random"
        compare = HEURISTICS_TO_COMPARE[1]   # expected "tcd"

        # Build numeric arrays from heur_results lists, replacing None by EPISODES_HEUR_COMPARE
        a_list = heur_results.get(baseline, [])
        b_list = heur_results.get(compare, [])
        a_num = np.array([x if x is not None else EPISODES_HEUR_COMPARE for x in a_list], dtype=float) if len(a_list) > 0 else np.array([], dtype=float)
        b_num = np.array([x if x is not None else EPISODES_HEUR_COMPARE for x in b_list], dtype=float) if len(b_list) > 0 else np.array([], dtype=float)

        if a_num.size == 0 or b_num.size == 0:
            print("\nNot enough heuristic data to compute Table 4.")
        else:
            mean_a = int(round(float(np.mean(a_num))))
            mean_b = int(round(float(np.mean(b_num))))
            sd_a = int(round(float(np.std(a_num, ddof=1)))) if a_num.size > 1 else 0
            sd_b = int(round(float(np.std(b_num, ddof=1)))) if b_num.size > 1 else 0

            # Percentage reduction of compare vs baseline: (compare - baseline)/baseline * 100
            reduction = (mean_b - mean_a) / float(mean_a) * 100.0
            reduction_rounded = round(reduction, 1)  # one decimal place

            # Print/record results
            print("\nTable 4 (Heuristic comparison)")
            print(f"Random init (baseline): mean={mean_a:,}, sd={sd_a:,}")
            print(f"TCD heuristic: mean={mean_b:,}, sd={sd_b:,}, reduction={reduction_rounded:+.1f}%")
    else:
        print("\nHeuristic comparison data not available to compute Table 4.")
        