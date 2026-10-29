"""
SQLite helper for storing and deduplicating sent tech news article URLs.
"""

import sqlite3
from typing import Any, Dict, Iterable, List, Set

DEFAULT_DB_PATH = "sent.db"


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite database schema if it doesn't already exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_articles (
                url TEXT PRIMARY KEY,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def get_sent_urls(db_path: str = DEFAULT_DB_PATH) -> Set[str]:
    """Retrieve the set of all URLs that have already been sent."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM sent_articles")
        rows = cursor.fetchall()
        return {row[0] for row in rows}


def filter_unsent_articles(
    articles: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Filter out articles whose URLs are already recorded in the database."""
    sent_urls = get_sent_urls(db_path)
    fresh: List[Dict[str, Any]] = []
    seen_in_batch: Set[str] = set()

    for article in articles:
        url = article.get("url")
        if not url:
            continue
        cleaned_url = url.strip()
        if cleaned_url not in sent_urls and cleaned_url not in seen_in_batch:
            seen_in_batch.add(cleaned_url)
            fresh.append(article)

    return fresh


def mark_urls_as_sent(
    urls: Iterable[str], db_path: str = DEFAULT_DB_PATH
) -> None:
    """Record a collection of URLs as sent in the database."""
    init_db(db_path)
    valid_urls = [url.strip() for url in urls if url and url.strip()]
    if not valid_urls:
        return

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO sent_articles (url) VALUES (?)",
            [(url,) for url in valid_urls],
        )
        conn.commit()

