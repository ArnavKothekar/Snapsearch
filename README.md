# SnapSearch

Snapsearch is an image search engine trained using the Caltech 101 image dataset. images from the Caltech 101 dataset are indexed in images/ and ran through **OpenCLIP ViT-B-32** (multimodal model).
Images and text are mapped in the same 512-dimensional embedding space, allowing comparisons to be made between text prompts and images. Embeddings are stored in SQLite, with the search method being a numpy cosine-similarity.
The search is ran over the whole collection in RAM. UI made using Streamlit.

### Rough Usage Flow

**Run 1 — first launch (indexing happens)**
1. You run streamlit run app.py.
2. config.py sets paths: images/ folder and snapsearch.db.
3. model.py loads the CLIP model into memory and picks the device (CUDA if available). This is the ~600 MB download on the very first run.
4. app.py asks the DB how many images are indexed (db.count_images()) and how many files are on disk. Since the DB is empty but images exist, it triggers auto-indexing (app.py:76).
5. indexer.index_images() starts:
  - _scan() walks images/ recursively and lists every image with its modified time.
  - It compares each file's mtime to the DB → all are new, so all get processed.
  - For each batch of 32: open with Pillow, downscale to ≤1024 px, convert to RGB.
  - embed_pil_images() runs each batch through CLIP on the GPU → a 512-number vector per image, normalized.
  - db.upsert_images() writes each vector + path + mtime into SQLite as a BLOB.
  - When done, delete_absent() removes DB rows for files that no longer exist.
6. The sidebar now shows "Indexed images: 9,144". The embeddings are saved, so this work never has to be repeated unless files change.
End of Run 1: the "knowledge" (embeddings) lives in snapsearch.db. The app is now ready to answer queries.

**Run 2 — second launch (a query happens)**
1. You run streamlit run app.py again.
2. app.py checks the DB → it already has 9,144 rows, so auto-indexing is skipped. (You can still click Re-index; it will compare mtimes and skip everything unchanged.)
3. You type a query like "red motorcycle" and click Search.
4. search.search_text() calls embed_text() → CLIP turns your sentence into a 512-number vector in the same space as the images.
5. search._load_matrix() reads all stored embeddings from SQLite into one numpy matrix (9,144 × 512).
6. _top_k() computes matrix @ query — one big dot product that scores every image's cosine similarity at once — then picks the highest top_k with argpartition.
7. It returns (relative_path, score) pairs, and app.py renders them as a grid of images with scores.
8. If you click Find similar on a result, it repeats steps 4–7 using that image file as the query instead of text (search_image_path).

- Run 1 = expensive, one-time work: convert images → vectors → save to SQLite.
- Run 2 = cheap, repeated work: convert one query → compare against saved vectors → show top matches.
The expensive CLIP image encoding only happens during indexing; every later query just encodes the single query and does fast numpy math against what's already stored.

Note: 
The indexer also deletes rows for vanished files (delete_absent),
Search re-reads the whole embedding matrix from SQLite on every query (no in-memory cache between queries).

### How to run it

Initial steps required (first time only)
1. Activate the venv: .\venv\Scripts\Activate.ps1
2. Install deps: pip install -r requirements.txt (use the CUDA build line for GPU)
3. Put images in images/
4. streamlit run app.py → auto-indexes, then you can search

First run downloads the CLIP weights (~600 MB) and auto-indexes images/. Add images then click Re-index to update.
Note docs/ and images/ are gitignored, so on GitHub the repo will show only the six .py files plus requirements.txt and .gitignore

### What does each file do?

**config.py** — Holds all tunable settings: image/database paths, the CLIP model name and checkpoint, thumbnail size, batch size, default result count, and valid image extensions. Every value can be overridden with an environment variable.
- BASE_DIR — project root path.
- IMAGES_DIR / DB_PATH — where images and the SQLite file live.
- MODEL_NAME / PRETRAINED — which CLIP model/weights to use.
- THUMBNAIL_SIZE / BATCH_SIZE / DEFAULT_TOP_K — indexing and search tuning.
- IMAGE_EXTENSIONS — accepted file types.
  
**db.py** — The SQLite layer. Defines the single images table (path, mtime, dimensions, embedding BLOB) and all read/write helpers.
- get_connection / init_db — open the DB and create the table.
- count_images — number of indexed rows.
- upsert_images — insert new or update changed embeddings.
- delete_absent — remove rows for missing files.
- fetch_images — load all paths and embeddings for search.
  
**model.py** — Loads CLIP once and turns text or images into normalized embeddings.
- get_device — pick CUDA or CPU.
- get_model — cached model/preprocess/tokenizer load.
- embed_text — text to vector.
- embed_pil_images — batch images to vectors.
- _run_under_autocast — fp16 speedup on GPU.
  
**indexer.py** — Scans images/, encodes new or changed files, and keeps the database in sync.
- _scan — list images with paths and mtimes.
- _load_thumbnail — open, downscale, and convert to RGB.
- index_images — skip unchanged, encode and store the rest, prune deleted, return a summary.
  
**search.py** — Encodes a query and ranks stored images by cosine similarity.
- _load_matrix — build a numpy matrix of all embeddings.
- _top_k — score everything and return the best matches.
- search_text — text query.
- search_image_path / search_image_bytes — reverse image search from a file or upload.
- search_image_by_query_array / resolve_path — helpers for precomputed vectors and path lookup.
  
**app.py** — The Streamlit UI that ties it all together: sidebar stats and re-index button, first-run auto-indexing, text and image search tabs, and a results grid with "find similar."
- images_on_disk — cached count of files on disk.
- run_indexing — run indexing with a progress bar.
- Sidebar — model/device info, counts, top-k slider, re-index.
- Auto-index guard — index automatically when the DB is empty.
- Search tabs — text input and image upload.
- Results — 3-column grid with scores and "find similar."

### Supporting files

requirements.txt	helps for installing required libraries using pip install -r requirements.txt torch, open-clip-torch, streamlit, pillow, numpy, tqdm (notes the CUDA install command)
.gitignore	ignores venv/, __pycache__/, snapsearch.db, images/, docs/
snapsearch.db	generated SQLite index (not committed)
images/	dataset, ~9,144 images in ~101 Caltech-101 category folders (not committed)
docs/	internal planning notes

### Use of CUDA GPU Acceleration

CUDA is used purely to accelerate the Pytorch/CLIP inference, with the logic being identical on CPU:
- Device is detected in model.py and if cuda is available it is used in open_clip.create_model_and_transforms(...)
- Every encode moves inputs to GPU
- Bare pip install torch on Windows installs the CPU-only build, so the CUDA build was installed from https://download.pytorch.org/whl/cu126.

- Text search tab and Image search upload tab (app.py:89)
- Results grid (3 columns) with scores and a Find similar button that re-queries with that image (app.py:136)
- images_on_disk() — cached (30s TTL) file count for the sidebar
