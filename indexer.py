from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from config import BATCH_SIZE, IMAGE_EXTENSIONS, IMAGES_DIR, THUMBNAIL_SIZE
import db
from model import embed_pil_images


def _scan():
    records = []
    if IMAGES_DIR.exists():
        for path in sorted(IMAGES_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rel_path = path.relative_to(IMAGES_DIR).as_posix()
                records.append((rel_path, str(path), path.stat().st_mtime))
    return records


def _load_thumbnail(path):
    img = Image.open(path)
    img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    return img.convert("RGB")


def index_images(progress=None):
    db.init_db()

    scanned = _scan()
    total = len(scanned)
    if total == 0:
        if progress:
            progress(1.0, "No images found")
        return {"total": 0, "added": 0, "updated": 0, "deleted": 0, "skipped": 0, "failed": 0, "errors": []}

    disk = {rel: (abs_path, mtime) for rel, abs_path, mtime in scanned}

    with db.get_connection() as conn:
        existing = {
            r["rel_path"]: r["file_mtime"]
            for r in conn.execute("SELECT rel_path, file_mtime FROM images")
        }

    to_process = [
        (rel, abs_path, mtime)
        for rel, (abs_path, mtime) in disk.items()
        if existing.get(rel) != mtime
    ]
    skipped = total - len(to_process)
    num_to_process = len(to_process)
    added = 0
    updated = 0
    failed = 0
    errors = []

    iterator = None
    if progress is None and num_to_process > 0:
        iterator = tqdm(total=num_to_process, desc="Indexing images", unit="img")

    processed = 0
    for start in range(0, num_to_process, BATCH_SIZE):
        batch = to_process[start : start + BATCH_SIZE]
        thumbs = []
        meta = []
        for rel, abs_path, mtime in batch:
            try:
                thumbs.append(_load_thumbnail(abs_path))
                meta.append((rel, abs_path, mtime))
            except Exception as exc:
                failed += 1
                errors.append(f"{rel}: {exc}")

        if thumbs:
            embeddings = embed_pil_images(thumbs)
            records = [
                (rel, abs_path, mtime, thumb.width, thumb.height, emb.astype(np.float32).tobytes())
                for (rel, abs_path, mtime), thumb, emb in zip(meta, thumbs, embeddings)
            ]
            db.upsert_images(records)
            for rel, _, _ in meta:
                if existing.get(rel) is None:
                    added += 1
                else:
                    updated += 1

        processed += len(batch)
        if progress:
            progress(min(processed / num_to_process, 1.0), f"Indexing {processed}/{num_to_process}")
        elif iterator is not None:
            iterator.update(len(batch))

    if iterator is not None:
        iterator.close()

    deleted = db.delete_absent(set(disk.keys()))

    if progress:
        progress(1.0, "Done")

    return {
        "total": total,
        "added": added,
        "updated": updated,
        "deleted": deleted,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
    }


if __name__ == "__main__":
    result = index_images()
    print(result)