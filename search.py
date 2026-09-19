import io

import numpy as np
from PIL import Image

from config import DEFAULT_TOP_K, IMAGES_DIR
import db
from model import embed_pil_images, embed_text


def _load_matrix():
    db.init_db()
    rows = db.fetch_images()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    dim = len(np.frombuffer(rows[0]["embedding"], dtype=np.float32))
    matrix = np.empty((len(rows), dim), dtype=np.float32)
    paths = []
    for i, row in enumerate(rows):
        matrix[i] = np.frombuffer(row["embedding"], dtype=np.float32)
        paths.append(row["rel_path"])
    return matrix, paths


def _top_k(query, top_k):
    matrix, paths = _load_matrix()
    if matrix.size == 0 or query.size == 0:
        return []
    scores = matrix @ query.astype(np.float32)
    k = min(top_k, len(paths))
    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(-scores[indices])]
    return [(paths[i], float(scores[i])) for i in indices]


def search_image_by_query_array(query, top_k=DEFAULT_TOP_K):
    return _top_k(query, top_k)


def search_text(query, top_k=DEFAULT_TOP_K):
    query_embedding = embed_text(query)[0]
    return _top_k(query_embedding, top_k)


def search_image_path(path, top_k=DEFAULT_TOP_K):
    query_embedding = embed_pil_images([Image.open(path).convert("RGB")])[0]
    return _top_k(query_embedding, top_k)


def search_image_bytes(data, top_k=DEFAULT_TOP_K):
    query_embedding = embed_pil_images([Image.open(io.BytesIO(data)).convert("RGB")])[0]
    return _top_k(query_embedding, top_k)


def resolve_path(rel_path):
    return IMAGES_DIR / rel_path