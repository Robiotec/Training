from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ROOT = Path("/home/robiotec/Documents/Entrenamientos/Training")
RESULT_DIR = ROOT / "result"


def read_field(args_path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(.*)$", re.MULTILINE)
    txt = args_path.read_text(encoding="utf-8", errors="ignore")
    match = pattern.search(txt)
    return match.group(1).strip() if match else ""


def classify_group(run_name: str, data_value: str) -> str:
    name = run_name.lower()
    data = data_value.lower()

    if "mixto" in name or "mixed" in name or "two_class" in name or "2class" in name:
        return "mixed"
    if "caja" in name:
        return "Caja"
    if "veta" in name:
        return "Vetas"

    if "solo_cajas" in data or "single_class_caja" in data or "data2.yaml" in data:
        return "Caja"
    if "solo_vetas" in data or "single_class_vetas" in data or "data.yaml" in data:
        return "Vetas"
    if "mixto" in data or "mixed" in data:
        return "mixed"
    return "unclassified"


def infer_split(data_value: str) -> str:
    data = data_value.replace("\\", "/")
    if not data:
        return "unknown_split"

    if data.endswith("data2.yaml"):
        return "historical_data2_yaml"
    if data.endswith("data.yaml"):
        return "historical_data_yaml"

    stem = Path(data).stem
    return stem if stem else "unknown_split"


def patch_save_dir(args_path: Path, new_save_dir: Path) -> None:
    txt = args_path.read_text(encoding="utf-8", errors="ignore")
    new_value = f"save_dir: {new_save_dir.as_posix()}"
    txt = re.sub(r"^save_dir:\s*.*$", new_value, txt, flags=re.MULTILINE)
    args_path.write_text(txt, encoding="utf-8")


def unique_target(base_target: Path) -> Path:
    if not base_target.exists():
        return base_target
    idx = 2
    while True:
        candidate = base_target.with_name(f"{base_target.name}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def cleanup_empty_dirs(base: Path) -> None:
    for p in sorted(base.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize result/ by class with clear run names")
    parser.add_argument("--apply", action="store_true", help="Ejecuta los movimientos (por defecto solo preview)")
    args = parser.parse_args()

    if not RESULT_DIR.exists():
        raise FileNotFoundError(f"No existe: {RESULT_DIR}")

    args_files = sorted(RESULT_DIR.rglob("args.yaml"))

    for args_path in args_files:
        run_dir = args_path.parent
        args_path = run_dir / "args.yaml"
        if not args_path.exists():
            continue

        run_name = read_field(args_path, "name") or run_dir.name
        data_value = read_field(args_path, "data")
        group = classify_group(run_name, data_value)
        split = infer_split(data_value)
        target_name = f"{split}__{run_dir.name}"
        target = unique_target(RESULT_DIR / group / target_name)

        print(f"{run_dir} -> {target}")

        if not args.apply:
            continue

        if run_dir.resolve() == target.resolve():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(target))

        new_args = target / "args.yaml"
        if new_args.exists():
            patch_save_dir(new_args, target)

    if args.apply:
        cleanup_empty_dirs(RESULT_DIR)


if __name__ == "__main__":
    main()
