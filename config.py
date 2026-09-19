import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = Path(os.environ.get("SNAPSEARCH_IMAGES", str(BASE_DIR / "images")))
DB_PATH = Path(os.environ.get("SNAPSEARCH_DB", str(BASE_DIR / "snapsearch.db")))

MODEL_NAME = os.environ.get("SNAPSEARCH_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("SNAPSEARCH_PRETRAINED", "laion2b_s34b_b79k")

THUMBNAIL_SIZE = int(os.environ.get("SNAPSEARCH_THUMBNAIL_SIZE", "1024"))
BATCH_SIZE = int(os.environ.get("SNAPSEARCH_BATCH_SIZE", "32"))
DEFAULT_TOP_K = int(os.environ.get("SNAPSEARCH_TOP_K", "9"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}