import sqlite3

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id         INTEGER PRIMARY KEY,
    rel_path   TEXT NOT NULL UNIQUE,
    abs_path   TEXT NOT NULL,
    file_mtime REAL NOT NULL,
    width      INTEGER,
    height     INTEGER,
    embedding  BLOB NOT NULL
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute(SCHEMA)


def count_images():
    init_db()
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()["c"]


def upsert_images(records):
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO images (rel_path, abs_path, file_mtime, width, height, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET
                abs_path   = excluded.abs_path,
                file_mtime = excluded.file_mtime,
                width      = excluded.width,
                height     = excluded.height,
                embedding  = excluded.embedding
            """,
            records,
        )


def delete_absent(keep_rel_paths):
    with get_connection() as conn:
        stored = {r["rel_path"] for r in conn.execute("SELECT rel_path FROM images")}
    vanished = sorted(stored - set(keep_rel_paths))
    if not vanished:
        return 0
    deleted = 0
    with get_connection() as conn:
        for i in range(0, len(vanished), 900):
            chunk = vanished[i : i + 900]
            placeholders = ",".join("?" * len(chunk))
            deleted += conn.execute(
                f"DELETE FROM images WHERE rel_path IN ({placeholders})", chunk
            ).rowcount
    return deleted


def fetch_images():
    with get_connection() as conn:
        return conn.execute("SELECT rel_path, embedding FROM images").fetchall()