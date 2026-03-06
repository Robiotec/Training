from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Reasigna todas las clases YOLO a 0")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("/home/robiotec/Documents/Entrenamientos/Training/data/splits"),
        help="Carpeta raíz que contiene archivos .txt de labels",
    )
    args = parser.parse_args()

    for txt in args.labels_dir.rglob("*.txt"):
        lines = txt.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            parts[0] = "0"
            new_lines.append(" ".join(parts))
        txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
