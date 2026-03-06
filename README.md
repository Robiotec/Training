## Project Structure

### Data
- `data/raw/minio_full/`: full historical dataset synchronized from MinIO.
- `data/curated/limpieza_caja/`: curated unsplit data for Caja workflow.
- `data/curated/limpieza_veta/`: curated unsplit data for Vetas workflow.
- `data/splits/`: versioned datasets with `train/val/test`.

### Dataset Configs
- `configs/vetas.yaml`
- `configs/caja.yaml`
- `configs/mixto.yaml` (`Caja=0`, `Veta=1`)

### Scripts
- `scripts/minio/download_from_minio.py`: download/sync from MinIO to `data/raw/minio_full/`.
- `scripts/minio/upload_to_minio.py`: upload local datasets to MinIO.
- `scripts/data/split_dataset.py`: generate YOLO datasets in `train/val/test`.
  - Also updates `configs/caja.yaml` and
    `configs/vetas.yaml` and `configs/mixto.yaml`
    train/val/test paths
    to the generated split (disable with `--no-update-training-configs`).
- `scripts/data/validate_labels_images.py`: visual validation with FiftyOne.
- `scripts/data/remap_labels_to_class0.py`: remap YOLO class id to `0`.

### Notebooks
- `notebooks/training.ipynb`: training, testing, and CUDA cleanup utilities.
- `notebooks/export_pt_to_engine.ipynb`: `.pt` to TensorRT `.engine` conversion.
- `notebooks/tuning/ray_tune.ipynb`: hyperparameter tuning.

### Training Results
- `result/<class>/<split>__<run>/`
- `<class>`: `Caja`, `Vetas`, `mixed`, `unclassified`.
- `<split>`: inferred split name, or a historical marker when `data.yaml`/`data2.yaml` was used.
- Organizer script: `scripts/results/organize_results.py`.
  - Preview: `python3 scripts/results/organize_results.py`
  - Apply: `python3 scripts/results/organize_results.py --apply`
- For new trainings, set `project` and `name` directly:
  - Example: `project=result/Vetas name=2026-03-06_vetas_v1__run_001`
