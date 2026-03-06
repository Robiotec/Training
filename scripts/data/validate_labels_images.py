import fiftyone as fo
import fiftyone.types as fot
import argparse
from pathlib import Path

DEFAULT_DATASET_DIR = Path("/home/robiotec/Documents/Entrenamientos/Training")
DEFAULT_YAML_PATH = Path("/home/robiotec/Documents/Entrenamientos/Training/configs/vetas.yaml")

parser = argparse.ArgumentParser(description="Validar un split YOLO a la vez en FiftyOne")
parser.add_argument(
    "--split",
    choices=["train", "val", "test"],
    default="train",
    help="Split a cargar (por defecto: train)",
)
parser.add_argument(
    "--dataset-dir",
    type=Path,
    default=DEFAULT_DATASET_DIR,
    help="Ruta base de dataset para resolver paths relativos del YAML",
)
parser.add_argument(
    "--yaml-path",
    type=Path,
    default=DEFAULT_YAML_PATH,
    help="Ruta al data.yaml de YOLO",
)
args = parser.parse_args()
dataset_name = f"vetas_{args.split}"

if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

dataset = fo.Dataset.from_dir(
    dataset_dir=str(args.dataset_dir),
    yaml_path=str(args.yaml_path),
    dataset_type=fot.YOLOv5Dataset,   # funciona con formato YOLO típico + data.yaml
    split=args.split,                 # carga solo un split por ejecución
    name=dataset_name,
)

session = fo.launch_app(dataset)  # abre UI en el navegador
session.wait()  # para que no se cierre
