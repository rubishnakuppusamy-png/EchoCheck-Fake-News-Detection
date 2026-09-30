"""
EchoCheck - SQLite Database Management Layer
Handles storing, retrieving, querying, and clearing prediction history.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional

DB_DIR = os.path.join(os.path.dirname(__file__), "database")
DB_PATH = os.path.join(DB_DIR, "predictions.db")

os.makedirs(DB_DIR, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database table and indices if they do not exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_text TEXT NOT NULL,
                article_snippet TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                fake_prob REAL NOT NULL,
                real_prob REAL NOT NULL,
                word_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_created 
            ON predictions(created_at DESC)
        """)
        conn.commit()


def save_prediction(
    article_text: str,
    predicted_label: str,
    confidence_score: float,
    fake_prob: float,
    real_prob: float,
    word_count: int
) -> int:
    """Inserts a new prediction record and returns its id."""
    # Create clean snippet for table preview
    clean_snippet = " ".join(article_text.split())
    if len(clean_snippet) > 160:
        clean_snippet = clean_snippet[:157] + "..."

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (article_text, article_snippet, predicted_label, confidence_score, fake_prob, real_prob, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            article_text,
            clean_snippet,
            predicted_label.upper(),
            float(confidence_score),
            float(fake_prob),
            float(real_prob),
            int(word_count)
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_predictions(search_query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves prediction history, optionally filtered by search text."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if search_query and search_query.strip():
            query_pattern = f"%{search_query.strip()}%"
            cursor.execute("""
                SELECT id, article_snippet, article_text, predicted_label, 
                       confidence_score, fake_prob, real_prob, word_count, 
                       strftime('%Y-%m-%d %H:%M:%S', created_at) as created_at
                FROM predictions
                WHERE article_text LIKE ? OR predicted_label LIKE ?
                ORDER BY id DESC
                LIMIT ?
            """, (query_pattern, query_pattern, limit))
        else:
            cursor.execute("""
                SELECT id, article_snippet, article_text, predicted_label, 
                       confidence_score, fake_prob, real_prob, word_count, 
                       strftime('%Y-%m-%d %H:%M:%S', created_at) as created_at
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_prediction_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single prediction by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, article_snippet, article_text, predicted_label, 
                   confidence_score, fake_prob, real_prob, word_count, 
                   strftime('%Y-%m-%d %H:%M:%S', created_at) as created_at
            FROM predictions
            WHERE id = ?
        """, (record_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_stats() -> Dict[str, Any]:
    """Calculates overall summary metrics for the history dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "total": 0,
                "real_count": 0,
                "fake_count": 0,
                "real_pct": 0,
                "fake_pct": 0,
                "avg_confidence": 0.0
            }

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 'REAL'")
        real_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_label = 'FAKE'")
        fake_count = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence_score) FROM predictions")
        avg_confidence = cursor.fetchone()[0] or 0.0

        return {
            "total": total,
            "real_count": real_count,
            "fake_count": fake_count,
            "real_pct": round((real_count / total) * 100, 1),
            "fake_pct": round((fake_count / total) * 100, 1),
            "avg_confidence": round(float(avg_confidence), 1)
        }


def clear_all_predictions() -> int:
    """Deletes all prediction records from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        deleted = cursor.rowcount
        conn.commit()
        return deleted


# Auto-initialize DB on import
init_db()
