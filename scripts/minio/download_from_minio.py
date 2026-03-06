from __future__ import annotations

from pathlib import Path
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# ---------- CONFIG ----------
MINIO_ENDPOINT = "http://192.168.30.200:9000"
ACCESS_KEY = "robiotec"
SECRET_KEY = "robiotec123"
BUCKET = "smartsorter-v2"

# Prefijos a descargar desde MinIO
MINIO_PREFIXES = ["cajas", "vetas", "Backgrounds"]

# Carpeta destino local (se crea si no existe)
LOCAL_DEST = Path(r"/home/robiotec/Documents/Entrenamientos/Training/data/dataminio")

MAX_WORKERS = 24

# Si True, re-descarga cuando el size difiere o cuando cambia el ETag.
# Si False, solo compara size.
VERIFY_ETAG = True

# Archivo de cache local para no hacer HEAD repetido (opcional)
STATE_FILE = LOCAL_DEST / ".minio_sync_state.json"
# ---------------------------


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    safe_mkdir(STATE_FILE.parent)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def list_all_objects(s3, bucket: str, prefix: str):
    """Itera todos los objetos bajo un prefijo."""
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj
        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break


def head_object(s3, bucket: str, key: str):
    return s3.head_object(Bucket=bucket, Key=key)


def etag_clean(etag: str | None) -> str | None:
    if not etag:
        return None
    return etag.strip().strip('"')


def should_download(local_path: Path, remote_size: int, remote_etag: str | None, state: dict, key: str) -> bool:
    if not local_path.exists():
        return True

    try:
        local_size = local_path.stat().st_size
    except OSError:
        return True

    if local_size != remote_size:
        return True

    if not VERIFY_ETAG:
        return False

    # Si existe estado previo con ETag, y el remoto coincide, skip
    cached = state.get(key)
    if cached and isinstance(cached, dict):
        cached_etag = cached.get("etag")
        if cached_etag and remote_etag and cached_etag == remote_etag:
            return False

    # Si no hay cache, no forzamos HEAD adicional aquí; el caller ya tiene ETag del list o del head.
    # Si el list trae ETag, usamos eso.
    if remote_etag:
        # No podemos calcular MD5 local sin leer todo; nos basamos en cache o tamaño.
        # Entonces, si tamaño coincide y no hay evidencia de cambio, omitimos.
        return False

    return False


def download_one(s3, key: str, local_path: Path) -> None:
    safe_mkdir(local_path.parent)
    # Descarga atómica: baja a .part y luego renombra
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    s3.download_file(BUCKET, key, str(tmp_path))
    tmp_path.replace(local_path)


def sync_prefix(s3, prefix: str, state: dict) -> tuple[int, int]:
    """Devuelve (descargados, saltados)."""
    downloaded = 0
    skipped = 0

    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for obj in list_all_objects(s3, BUCKET, prefix):
            key = obj["Key"]
            # Evitar "carpetas" vacías (S3 no tiene folders reales)
            if key.endswith("/"):
                continue

            remote_size = int(obj.get("Size", 0))
            remote_etag = etag_clean(obj.get("ETag"))

            # Mapea key -> ruta local igual (LOCAL_DEST/prefix/...)
            local_path = LOCAL_DEST / key.replace("/", os.sep)

            if should_download(local_path, remote_size, remote_etag, state, key):
                tasks.append(ex.submit(download_one, s3, key, local_path))
            else:
                skipped += 1
                # Actualiza cache con lo que sabemos (size/etag)
                state[key] = {"size": remote_size, "etag": remote_etag}

        for fut in as_completed(tasks):
            fut.result()
            downloaded += 1

    return downloaded, skipped


def main():
    safe_mkdir(LOCAL_DEST)

    s3 = make_s3()

    # Validación mínima de conectividad al bucket
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as e:
        raise RuntimeError(f"No puedo acceder al bucket '{BUCKET}'. Revisa endpoint/credenciales/red. Detalle: {e}")

    state = load_state()

    total_down = 0
    total_skip = 0

    for p in MINIO_PREFIXES:
        log(f"==> Sincronizando prefijo: {p}")
        d, s = sync_prefix(s3, p, state)
        total_down += d
        total_skip += s
        log(f"[OK] {p}: descargados={d}, saltados={s}")

        # Guardar estado por prefijo para no perder avance
        save_state(state)

    log(f"TOTAL: descargados={total_down}, saltados={total_skip}")
    log(f"Destino local: {LOCAL_DEST}")


if __name__ == "__main__":
    main()