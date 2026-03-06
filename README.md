# Training Pipeline for Caja/Vetas Detection

This repository contains the full dataset-to-model workflow for YOLO training in the Smart Sorter context.
It covers:

- MinIO sync (download/upload)
- Dataset curation and split generation
- Training and validation workflows
- PT to TensorRT engine export
- Result organization and traceability

The project is structured for reproducible experiments, with explicit split versions and class-specific outputs (`Caja`, `Vetas`, `mixed`).

## Architecture Overview

```text
                 +--------------------------------------+
                 |            MinIO Bucket              |
                 |   smartsorter-v2/{cajas,vetas,BG}   |
                 +------------------+-------------------+
                                    |
                                    v
                    +---------------+----------------+
                    | scripts/minio/download_*       |
                    | -> data/raw/minio_full/        |
                    +---------------+----------------+
                                    |
                                    v
                    +---------------+----------------+
                    | data/curated/{limpieza_*}/     |
                    | Manual curation and cleanup    |
                    +---------------+----------------+
                                    |
                                    v
                    +---------------+----------------+
                    | scripts/data/split_dataset.py  |
                    | -> data/splits/<version>/      |
                    | -> updates configs/*.yaml      |
                    +---------------+----------------+
                                    |
                                    v
                  +-----------------+------------------+
                  | notebooks/training.ipynb           |
                  | train / val / predict / gpu clean  |
                  +-----------------+------------------+
                                    |
                                    v
                  +-----------------+------------------+
                  | result/{Caja|Vetas|mixed}/...      |
                  | versioned runs and metrics         |
                  +-----------------+------------------+
                                    |
                                    v
                  +-----------------+------------------+
                  | notebooks/export_pt_to_engine.ipynb|
                  | .pt -> TensorRT .engine            |
                  +------------------------------------+
```

## Repository Structure

```text
Training/
├── configs/
│   ├── caja.yaml
│   ├── vetas.yaml
│   └── mixto.yaml
│
├── data/
│   ├── raw/minio_full/                 # Full historical sync from MinIO
│   ├── curated/
│   │   ├── limpieza_caja/              # Curated unsplit source for Caja workflows
│   │   └── limpieza_veta/              # Curated unsplit source for Vetas workflows
│   └── splits/                         # Versioned split outputs (train/val/test)
│
├── model/
│   ├── yolo26n.pt                      # Base training model
│   └── one_class.engine                # TensorRT artifact (if generated)
│
├── notebooks/
│   ├── training.ipynb                  # Training/validation/predict workflow
│   ├── export_pt_to_engine.ipynb       # PT -> TensorRT export workflow
│   └── tuning/ray_tune.ipynb           # Hyperparameter tuning
│
├── result/
│   ├── Caja/
│   └── Vetas/
│
├── scripts/
│   ├── data/
│   │   ├── split_dataset.py
│   │   ├── remap_labels_to_class0.py
│   │   └── validate_labels_images.py
│   ├── minio/
│   │   ├── download_from_minio.py
│   │   └── upload_to_minio.py
│   └── results/
│       ├── organize_results.py
│       └── README.txt
│
├── pyproject.toml
├── uv.lock
└── README.md
```

## Environment Setup (UV)

### 1. Sync dependencies

```bash
uv sync
```

### 2. (Optional) Activate venv

```bash
source .venv/bin/activate
```

### 3. Verify key tooling

```bash
python3 --version
yolo --help
```

## Configuration Model

The training configs are centralized in:

- `configs/caja.yaml`
- `configs/vetas.yaml`
- `configs/mixto.yaml`

These YAML files define `train`, `val`, `test`, `nc`, and `names`.

Important behavior:

- `scripts/data/split_dataset.py` can automatically rewrite these YAMLs to point to the latest generated split.
- This allows notebooks and training commands to remain stable while data versions change.

## End-to-End Workflow

### Step 1. Sync raw data from MinIO

```bash
python3 scripts/minio/download_from_minio.py
```

Default destination:

- `data/raw/minio_full/`

### Step 2. Curate data manually

Use the curated folders:

- `data/curated/limpieza_caja/`
- `data/curated/limpieza_veta/`

### Step 3. Build train/val/test splits

Example (default mode is `solo_vetas`):

```bash
python3 scripts/data/split_dataset.py \
  --base-dir data/curated/limpieza_veta \
  --out-dir data/splits \
  --dataset-prefix 2026-03-06 \
  --modes solo_vetas
```

Generate multiple modes in one pass:

```bash
python3 scripts/data/split_dataset.py \
  --base-dir data/curated/limpieza_veta \
  --out-dir data/splits \
  --dataset-prefix 2026-03-06 \
  --modes solo_cajas solo_vetas mixto
```

Do not update training YAMLs automatically:

```bash
python3 scripts/data/split_dataset.py --no-update-training-configs
```

### Step 4. Validate split labels/images in FiftyOne

```bash
python3 scripts/data/validate_labels_images.py \
  --yaml-path configs/vetas.yaml \
  --split train
```

### Step 5. (Optional) Remap labels to class 0

```bash
python3 scripts/data/remap_labels_to_class0.py \
  --labels-dir data/splits
```

### Step 6. Train model

Use:

- `notebooks/training.ipynb`

Notebook sections are ordered to:

1. Set environment
2. Define shared paths/defaults
3. Register utility functions
4. Quick GPU cleanup
5. Configure one run
6. Train
7. Validate
8. Predict (optional)

### Step 7. Export PT to TensorRT

Use:

- `notebooks/export_pt_to_engine.ipynb`

This notebook is organized for:

1. GPU/CUDA/TensorRT checks
2. PT model + YAML config setup
3. `.pt -> .engine` export
4. Optional sanity inference and speed check

## Result Organization

Expected structure:

- `result/<class>/<split>__<run>/`

Where:

- `<class>`: `Caja`, `Vetas`, `mixed`, `unclassified`
- `<split>`: inferred from training `data` yaml
- `<run>`: run folder name

Reorganize legacy runs:

```bash
python3 scripts/results/organize_results.py           # preview
python3 scripts/results/organize_results.py --apply   # apply changes
```

## Naming Convention (Recommended)

For split folders:

- `YYYY-MM-DD_<scope>_vN`
- Example: `2026-03-06_solo_vetas_v1`

For training runs:

- `<split_tag>__run_XXX`
- Example: `2026-03-06_solo_vetas_v1__run_001`

For result path grouping:

- `result/Vetas/<split_tag>__run_001`
- `result/Caja/<split_tag>__run_001`

## Common Commands

```bash
# Sync env
uv sync

# Build split(s)
python3 scripts/data/split_dataset.py --base-dir data/curated/limpieza_veta --out-dir data/splits --dataset-prefix 2026-03-06 --modes solo_vetas

# Validate split in FiftyOne
python3 scripts/data/validate_labels_images.py --yaml-path configs/vetas.yaml --split val

# Organize results
python3 scripts/results/organize_results.py --apply
```

## Operational Notes

- Some MinIO scripts currently contain hardcoded endpoint/credentials and absolute paths. Consider moving them to environment variables before production sharing.
- Keep large artifacts (`data/`, `result/`, `*.pt`, `*.engine`) out of Git unless strictly needed.
- Treat `data/raw/minio_full/` as immutable historical source and run curation in `data/curated/`.

## Minimal Run Checklist

1. `uv sync`
2. Sync/download raw data (if needed)
3. Curate data under `data/curated/`
4. Run `split_dataset.py`
5. Confirm `configs/*.yaml` paths are correct
6. Train from `notebooks/training.ipynb`
7. Validate and store run in `result/<class>/...`
8. Export to TensorRT if deployment requires `.engine`
