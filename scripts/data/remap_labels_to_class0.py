from pathlib import Path

labels_dir = Path("/home/robiotec/Documents/Entrenamientos/Training/data/DS_ONLY_VETAS/test/labels")  # carpeta con tus .txt

for txt in labels_dir.rglob("*.txt"):
    lines = txt.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        # si solo hay vetas, fuerza todo a 0
        parts[0] = "0"
        new_lines.append(" ".join(parts))
    txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
