from pathlib import Path
import boto3
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
MINIO_ENDPOINT = "http://192.168.30.200:9000"
ACCESS_KEY = "robiotec"
SECRET_KEY = "robiotec123"
BUCKET = "smartsorter-v2"

PARENT_FOLDER = Path(r"C:\Users\yuchu\Downloads\data")  # carpeta padre con chuteX, chuteX1, ...

LOCAL_TO_MINIO = {
    "Caja": "cajas",
    "Veta": "vetas",
    "BG": "Backgrounds",
}

MAX_WORKERS = 24
# ----------------------------


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def upload_one(s3, file_path: Path, base_folder: Path, minio_prefix: str, chute_name: str):
    rel = file_path.relative_to(base_folder).as_posix()  # mantiene images/labels/...
    key = f"{minio_prefix}/{chute_name}/{rel}"
    s3.upload_file(str(file_path), BUCKET, key)
    return key


def process_chute(s3, chute_dir: Path):
    chute_name = chute_dir.name
    total = 0
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for local_class, minio_prefix in LOCAL_TO_MINIO.items():
            class_dir = chute_dir / local_class
            if not class_dir.exists() or not class_dir.is_dir():
                continue  # si no existe Caja/Veta/BG en este chute, se omite

            # opcional: solo images/labels (si quieres ser estricto)
            # for sub in ["images", "labels"]:
            #     subdir = class_dir / sub
            #     if subdir.exists():
            #         for f in subdir.rglob("*"):
            #             if f.is_file():
            #                 futures.append(ex.submit(upload_one, s3, f, class_dir, minio_prefix, chute_name))

            # general: sube TODO lo que esté dentro de Caja/Veta/BG
            for f in class_dir.rglob("*"):
                if f.is_file():
                    futures.append(ex.submit(upload_one, s3, f, class_dir, minio_prefix, chute_name))

        for fut in as_completed(futures):
            _ = fut.result()
            total += 1

    print(f"[OK] {chute_name}: {total} archivos subidos.")
    return total


def main():
    if not PARENT_FOLDER.exists():
        raise FileNotFoundError(f"No existe: {PARENT_FOLDER}")

    s3 = make_s3()

    chutes = [p for p in PARENT_FOLDER.iterdir() if p.is_dir()]
    if not chutes:
        print(f"No hay subcarpetas dentro de: {PARENT_FOLDER}")
        return

    grand_total = 0
    for chute in chutes:
        grand_total += process_chute(s3, chute)

    print(f"TOTAL: {grand_total} archivos subidos.")


if __name__ == "__main__":
    main()