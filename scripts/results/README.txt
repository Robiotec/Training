organize_results.py
===================

Purpose
- Reorganize training runs inside `result/` so they are easy to find by class and split.

Output structure
- `result/<class>/<split>__<run>/`

Where:
- `<class>` can be: `Caja`, `Vetas`, `mixed`, `unclassified`
- `<split>` is inferred from `args.yaml` (`data` field):
  - `data.yaml`  -> `historical_data_yaml`
  - `data2.yaml` -> `historical_data2_yaml`
  - any other yaml path -> yaml filename (without extension)

How it works
1. Finds run folders in `result/` that contain `args.yaml`.
2. Reads `name`, `data`, and `save_dir` from `args.yaml`.
3. Builds target path using class and split.
4. With `--apply`, moves the folder and updates `save_dir` in `args.yaml`.

Usage
- Preview (no changes):
  `python3 scripts/results/organize_results.py`

- Apply changes:
  `python3 scripts/results/organize_results.py --apply`

Notes
- If target already exists, suffix `_2`, `_3`, etc. is added.
- If class cannot be inferred confidently, run goes to `unclassified`.
