from pathlib import Path

import streamlit as st

import search
from config import DEFAULT_TOP_K, IMAGE_EXTENSIONS, IMAGES_DIR, MODEL_NAME
import db
from indexer import index_images
from model import get_device

st.set_page_config(
    page_title="SnapSearch",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

device = get_device()


@st.cache_data(ttl=30, show_spinner=False)
def images_on_disk():
    if not IMAGES_DIR.exists():
        return 0
    return sum(
        1
        for p in IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_indexing():
    bar = st.sidebar.progress(0, text="Indexing images...")
    result = index_images(progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
    return result


# ---------------- Sidebar ----------------
st.sidebar.title("SnapSearch")
st.sidebar.caption(f"Model: **{MODEL_NAME}**  ·  Device: **{device}**")

indexed = db.count_images()
disk_count = images_on_disk()
st.sidebar.metric("Indexed images", f"{indexed:,}")
st.sidebar.caption(f"{disk_count:,} images on disk" if disk_count else "no images found in images/")

top_k = st.sidebar.slider(
    "Results shown", min_value=1, max_value=50, value=DEFAULT_TOP_K
)

if st.sidebar.button("Re-index images", type="primary", use_container_width=True):
    with st.spinner("Indexing images... (first run downloads the CLIP model)"):
        result = run_indexing()
    added = result["added"]
    updated = result["updated"]
    deleted = result["deleted"]
    skipped = result["skipped"]
    failed = result["failed"]
    st.sidebar.success(
        f"Done — {added} added, {updated} updated, {deleted} removed, "
        f"{skipped} unchanged, {failed} failed."
    )
    if result["errors"]:
        with st.sidebar.expander(f"{len(result['errors'])} error(s)"):
            for err in result["errors"][:20]:
                st.code(err)
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Tip: drop more images into `images/`, then click Re-index.")

# ---------------- Initial auto-index ----------------
if "auto_index_attempted" not in st.session_state:
    st.session_state.auto_index_attempted = False

if indexed == 0 and disk_count > 0 and not st.session_state.auto_index_attempted:
    st.session_state.auto_index_attempted = True
    with st.spinner("Indexing your images for the first time... (downloads the model once)"):
        result = run_indexing()
    st.success(
        f"Initial index complete: {result['total']:,} images scanned, "
        f"{result['added']:,} added, {result['failed']} failed."
    )

# ---------------- Main ----------------
st.title("SnapSearch")
st.caption("Semantic search over your images with CLIP + SQLite.")

tab_text, tab_image = st.tabs(["Text search", "Image search"])

with tab_text:
    with st.form("text_search_form", clear_on_submit=False):
        query = st.text_input(
            "Describe what you're looking for",
            placeholder="e.g. a red motorcycle on a road",
        )
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and query.strip():
        with st.spinner("Searching..."):
            results = search.search_text(query.strip(), top_k=top_k)
        st.session_state.results = results
        st.session_state.results_label = f'Text: "{query.strip()}"'
        st.session_state.results_top_k = top_k

with tab_image:
    uploaded = st.file_uploader(
        "Upload an image to find similar ones",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )
    if uploaded is not None:
        with st.spinner("Searching..."):
            results = search.search_image_bytes(uploaded.getvalue(), top_k=top_k)
        st.session_state.results = results
        st.session_state.results_label = f"Similar to: {uploaded.name}"
        st.session_state.results_top_k = top_k

pending_similar = st.session_state.pop("pending_similar", None)
if pending_similar:
    with st.spinner("Searching for similar images..."):
        results = search.search_image_path(pending_similar, top_k=top_k)
    st.session_state.results = results
    st.session_state.results_label = f"Similar to: {Path(pending_similar).name}"

st.subheader("Results")
results = st.session_state.get("results")
if results is None:
    st.info("Enter a text query or upload an image to search.")
else:
    if disk_count == 0:
        st.warning("Images found in storage are indexed. Nothing to search yet.")
    if not results:
        st.warning("No matches returned. Try a different query, or index more images.")
    else:
        st.caption(f"{st.session_state.get('results_label', '')} — {len(results)} result(s)")
        columns = st.columns(3)
        for i, (rel_path, score) in enumerate(results):
            with columns[i % 3]:
                full_path = search.resolve_path(rel_path)
                st.image(str(full_path), caption=f"{rel_path}  ·  {score:.3f}", width="stretch")
                if st.button("Find similar", key=f"similar_{i}", use_container_width=True):
                    st.session_state.pending_similar = str(full_path)
                    st.rerun()