from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"


@dataclass(frozen=True)
class Sample:
    img_path: Path
    label_lines: List[str]  # ya convertidas (puede ser [])


def find_image_label_pairs(root: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Busca recursivamente carpetas que contengan 'images' y 'labels',
    y retorna pares (imagen, label_path o None si no existe).
    """
    pairs: List[Tuple[Path, Optional[Path]]] = []

    # Detecta cualquier carpeta ".../images" dentro del árbol
    for images_dir in root.rglob("images"):
        if not images_dir.is_dir():
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            continue

        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                lab = labels_dir / f"{img.stem}.txt"
                pairs.append((img, lab if lab.exists() else None))

    return pairs


def read_yolo_label(label_path: Optional[Path]) -> List[str]:
    if label_path is None:
        return []
    try:
        txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return []
    if not txt:
        return []
    # Normaliza líneas no vacías
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def convert_labels(
    raw_lines: List[str],
    mode: str,
    source_kind: str,
) -> List[str]:
    """
    mode:
      - "solo_cajas": 1 clase (0=caja)
      - "solo_vetas": 1 clase (0=veta)
      - "mixto": 2 clases (0=caja, 1=veta)

    source_kind:
      - "cajas" | "vetas" | "backgrounds"
    """
    # Backgrounds siempre vacío
    if source_kind == "backgrounds":
        return []

    # En los modos "solo_*", solo la carpeta target conserva bboxes; lo demás se vuelve BG.
    if mode == "solo_cajas" and source_kind != "cajas":
        return []
    if mode == "solo_vetas" and source_kind != "vetas":
        return []

    out: List[str] = []

    for ln in raw_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue

        # Asumimos que dentro de:
        # - cajas/: caja=0 (si hay algo más, lo ignoramos)
        # - vetas/: veta=1 (si tu dataset trae 0/1), pero como es "solo_vetas" o "mixto",
        #   vamos a remapear robusto según carpeta.
        if mode in {"solo_cajas", "solo_vetas"}:
            # 1-clase: todo lo que venga en la carpeta objetivo lo pasamos a clase 0
            # (si hay líneas de otra clase dentro de esa carpeta, las descartamos por seguridad)
            if mode == "solo_cajas":
                # conservamos solo si cls == 0
                if cls != 0:
                    continue
                new_cls = 0
            else:
                # conservamos solo si cls == 1 (típico: veta=1)
                if cls != 1:
                    continue
                new_cls = 0

            out.append(" ".join([str(new_cls)] + parts[1:5]))
            continue

        if mode == "mixto":
            # Remapeo por carpeta (más confiable que confiar en el índice que venga):
            # cajas -> 0, vetas -> 1
            new_cls = 0 if source_kind == "cajas" else 1
            # Si quieres filtrar por cls original también, descomenta:
            # if source_kind == "cajas" and cls != 0: continue
            # if source_kind == "vetas" and cls != 1: continue
            out.append(" ".join([str(new_cls)] + parts[1:5]))
            continue

    return out


def split_indices(n: int, train: float, val: float, test: float) -> Tuple[List[int], List[int], List[int]]:
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("train+val+test debe sumar 1.0")

    idx = list(range(n))
    # ya vienen barajados afuera, pero igual
    # random.shuffle(idx)

    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    assert len(test_idx) == n_test
    return train_idx, val_idx, test_idx


def split_values(values: List[int], train: float, val: float, test: float) -> Tuple[List[int], List[int], List[int]]:
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("train+val+test debe sumar 1.0")

    n = len(values)
    n_train = int(n * train)
    n_val = int(n * val)
    tr = values[:n_train]
    va = values[n_train:n_train + n_val]
    te = values[n_train + n_val:]
    return tr, va, te


def stratified_split_indices_by_labels(
    samples: List[Sample],
    train: float,
    val: float,
    test: float,
) -> Tuple[List[int], List[int], List[int]]:
    pos_idx = [i for i, s in enumerate(samples) if s.label_lines]
    bg_idx = [i for i, s in enumerate(samples) if not s.label_lines]

    random.shuffle(pos_idx)
    random.shuffle(bg_idx)

    tr_pos, va_pos, te_pos = split_values(pos_idx, train, val, test)
    tr_bg, va_bg, te_bg = split_values(bg_idx, train, val, test)

    tr_idx = tr_pos + tr_bg
    va_idx = va_pos + va_bg
    te_idx = te_pos + te_bg

    random.shuffle(tr_idx)
    random.shuffle(va_idx)
    random.shuffle(te_idx)
    return tr_idx, va_idx, te_idx


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def format_split_summary(
    dataset_name: str,
    total: int,
    train_total: int,
    val_total: int,
    test_total: int,
    tr_pos: int,
    tr_bg: int,
    va_pos: int,
    va_bg: int,
    te_pos: int,
    te_bg: int,
) -> str:
    return "\n".join([
        f"dataset: {dataset_name}",
        f"total_images: {total}",
        f"train_total: {train_total}",
        f"val_total: {val_total}",
        f"test_total: {test_total}",
        f"train: caja={tr_pos} bg={tr_bg}",
        f"val: caja={va_pos} bg={va_bg}",
        f"test: caja={te_pos} bg={te_bg}",
    ]) + "\n"


def write_training_config(config_path: Path, split_dataset_dir: Path, class_names: List[str]) -> None:
    try:
        split_rel = split_dataset_dir.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        split_rel = split_dataset_dir.resolve()

    train_path = (split_rel / "train").as_posix()
    val_path = (split_rel / "val").as_posix()
    test_path = (split_rel / "test").as_posix()

    lines = [
        f"train: {train_path}",
        f"val: {val_path}",
        f"test: {test_path}",
        "# Number of classes",
        f"nc: {len(class_names)}",
        "# Class names",
        f"names: {class_names}",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset(
    samples: List[Sample],
    out_root: Path,
    dataset_name: str,
    class_names: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_labels: bool = True,
) -> Path:
    random.seed(seed)
    random.shuffle(samples)

    ds_dir = out_root / dataset_name
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ensure_dir(ds_dir)

    # Estructura YOLO
    for split in ("train", "val", "test"):
        ensure_dir(ds_dir / split / "images")
        ensure_dir(ds_dir / split / "labels")

    if stratify_by_labels:
        tr_idx, va_idx, te_idx = stratified_split_indices_by_labels(samples, train_ratio, val_ratio, test_ratio)
    else:
        tr_idx, va_idx, te_idx = split_indices(len(samples), train_ratio, val_ratio, test_ratio)

    def copy_split(indices: List[int], split: str, start_counter: int) -> int:
        counter = start_counter
        for i in indices:
            s = samples[i]
            counter += 1
            new_stem = f"IMG{counter:06d}"
            new_img = ds_dir / split / "images" / f"{new_stem}{s.img_path.suffix.lower()}"
            new_lbl = ds_dir / split / "labels" / f"{new_stem}.txt"

            shutil.copy2(s.img_path, new_img)
            new_lbl.write_text("\n".join(s.label_lines) + ("\n" if s.label_lines else ""), encoding="utf-8")
        return counter

    c = 0
    c = copy_split(tr_idx, "train", c)
    c = copy_split(va_idx, "val", c)
    c = copy_split(te_idx, "test", c)

    # data.yaml
    yaml = [
        f"path: {ds_dir.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(class_names)}",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(class_names)]

    (ds_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    def count_pos(indices: List[int]) -> int:
        return sum(1 for i in indices if samples[i].label_lines)

    tr_pos = count_pos(tr_idx)
    va_pos = count_pos(va_idx)
    te_pos = count_pos(te_idx)
    tr_bg = len(tr_idx) - tr_pos
    va_bg = len(va_idx) - va_pos
    te_bg = len(te_idx) - te_pos

    print(f"[OK] {dataset_name}: {len(samples)} imgs | train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    print(f"     train: caja={tr_pos} bg={tr_bg} | val: caja={va_pos} bg={va_bg} | test: caja={te_pos} bg={te_bg}")

    summary_txt = format_split_summary(
        dataset_name=dataset_name,
        total=len(samples),
        train_total=len(tr_idx),
        val_total=len(va_idx),
        test_total=len(te_idx),
        tr_pos=tr_pos,
        tr_bg=tr_bg,
        va_pos=va_pos,
        va_bg=va_bg,
        te_pos=te_pos,
        te_bg=te_bg,
    )
    (ds_dir / "split_summary.txt").write_text(summary_txt, encoding="utf-8")
    return ds_dir


def build_samples(base_dir: Path, mode: str) -> List[Sample]:
    # Mapea kind -> posibles raíces. Si faltan nombres estándar, intenta autodetectar por nombre.
    roots = {
        "cajas": [base_dir / "cajas"],
        "vetas": [base_dir / "vetas"],
        "backgrounds": [base_dir / "Backgrounds", base_dir / "backgrounds", base_dir / "background"],
    }

    if base_dir.exists():
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if "caja" in name:
                roots["cajas"].append(child)
            elif "veta" in name:
                roots["vetas"].append(child)
            elif name in {"bg", "bgs", "background", "backgrounds"}:
                roots["backgrounds"].append(child)

    all_samples: List[Sample] = []

    for kind, candidates in roots.items():
        seen_roots = set()
        found_any = False
        for root in candidates:
            root = root.resolve()
            if root in seen_roots:
                continue
            seen_roots.add(root)
            if not root.exists():
                continue

            found_any = True
            pairs = find_image_label_pairs(root)
            for img_path, lbl_path in pairs:
                raw = read_yolo_label(lbl_path)
                conv = convert_labels(raw, mode=mode, source_kind=kind)
                all_samples.append(Sample(img_path=img_path, label_lines=conv))

        if not found_any:
            print(f"[WARN] No se encontró raíz para kind='{kind}' dentro de {base_dir}")

    return all_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split de dataset YOLO por modo")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/home/robiotec/Documents/Entrenamientos/Training/data/curated/limpieza_veta"),
        help="Carpeta base no-spliteada (ej. limpieza_veta o limpieza_caja)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/robiotec/Documents/Entrenamientos/Training/data/splits"),
        help="Carpeta raíz donde se crearán los datasets spliteados",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="2026-03-06",
        help="Prefijo para versionado (ej. 2026-03-06)",
    )
    parser.add_argument(
        "--no-update-training-configs",
        action="store_true",
        help="No actualiza los YAML de entrenamiento",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["solo_cajas", "solo_vetas", "mixto"],
        default=["solo_vetas"],
        help="Modo(s) a generar. Ejemplo: --modes solo_cajas mixto",
    )
    args = parser.parse_args()
    ensure_dir(args.out_dir)

    mode_cfg = {
        "solo_cajas": {"dataset_name": f"{args.dataset_prefix}_solo_cajas_v1", "class_names": ["Caja"]},
        "solo_vetas": {"dataset_name": f"{args.dataset_prefix}_solo_vetas_v1", "class_names": ["Vetas"]},
        "mixto": {"dataset_name": f"{args.dataset_prefix}_mixto_v1", "class_names": ["Caja", "Veta"]},
    }

    global_summary: List[str] = []
    for mode in args.modes:
        cfg = mode_cfg[mode]
        samples = build_samples(args.base_dir, mode=mode)
        ds_dir = write_dataset(
            samples,
            args.out_dir,
            dataset_name=cfg["dataset_name"],
            class_names=cfg["class_names"],
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            seed=42,
        )
        if not args.no_update_training_configs:
            if mode == "solo_cajas":
                write_training_config(
                    DATASET_CONFIG_DIR / "caja.yaml",
                    ds_dir,
                    cfg["class_names"],
                )
            elif mode == "solo_vetas":
                write_training_config(
                    DATASET_CONFIG_DIR / "vetas.yaml",
                    ds_dir,
                    cfg["class_names"],
                )
            elif mode == "mixto":
                write_training_config(
                    DATASET_CONFIG_DIR / "mixto.yaml",
                    ds_dir,
                    cfg["class_names"],
                )
        global_summary.append((ds_dir / "split_summary.txt").read_text(encoding="utf-8").strip())

    (args.out_dir / f"{args.dataset_prefix}_split_summary_global.txt").write_text(
        "\n\n".join(global_summary) + "\n",
        encoding="utf-8",
    )

    print(f"\nListo. Revisa en: {args.out_dir.resolve()}")
