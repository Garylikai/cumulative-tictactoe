# cumulative-tictactoe

Official implementation for the manuscript **"Combinatorial Game Theory and Reinforcement Learning in Cumulative Tic-Tac-Toe via Evaluation Functions"**.  
Contains the game engine, TD-learning agents, and scripts used for statistical reproduction.

This repository contains:
- `cumulative_tictactoe_core.py` — game engine, `State`, `Judger`, `Player` (TD), and `HumanPlayer`.
- `cumulative_tictactoe_run.py` — single-run training / evaluation / interactive human vs AI entry points.
- `cumulative_tictactoe_hyper_run.py` — hyperparameter grid-search, reproducibility driver, and statistical analysis helpers.

## Requirements

Minimal third-party dependencies: numpy>=1.21, scipy>=1.7

Recommended Python: **3.8+**

## Purpose of the runner scripts

- `cumulative_tictactoe_run.py`  
  Use for **single-run training, evaluation, and interactive human vs AI play**. Typical uses:
  - Run one training session via `train()` and produce two policy files (one per player).
  - Evaluate saved policies in `compete()` (AI vs AI).
  - Launch an interactive human vs AI session via `play()`.

  Best for individual experiments, debugging, and demonstrations.

- `cumulative_tictactoe_hyper_run.py`  
  Use for **systematic experiments and reproducibility**:
  - Grid-search over seeds, `epsilon`, and `step_size`.
  - Record `episodes_to_converge`, elapsed time, and per-run metadata.
  - Perform the two-heuristic comparison and (optionally) run statistical tests (Shapiro, Levene, $t$-test).
  - Saves aggregate results to `hyper_grid_results.csv` and `hyper_grid_results.json` by default.

  Intended to reproduce the data reported in the manuscript.

## Quick usage examples

Run a short training demo (1000 episodes):
```
python -c "from cumulative_tictactoe_run import train; train(1000, seed=42)"
```

Run the hyperparameter driver with default constants:
```
python cumulative_tictactoe_hyper_run.py
```

For quick local tests, edit the top-of-file constants in `cumulative_tictactoe_hyper_run.py`:

- `EPISODES_GRID` — episodes per grid configuration (set to `1e3` or `1e4` for quick runs)
- `SEEDS`, `EPSILONS`, `STEP_SIZES` — grid values
- `HEURISTIC_FOR_GRID` — `"zero"`, `"tcd"`, or `"random"`

## Outputs produced

- Default policy files: `policy_first.bin`, `policy_second.bin` (overwritten by subsequent runs unless renamed).
- Hyper-run manifests: `hyper_grid_results.csv`, `hyper_grid_results.json`.
- Heuristic-comparison: `heuristic_compare_results.json`.

## License & citation

The code is released under the MIT License. If you use this code or data in this published work, please cite the associated manuscript.

## Contact

Author: Kai Li — [kai.li@stonybrook.edu](mailto:kai.li@stonybrook.edu)
