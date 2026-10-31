"""
SQLite helper for storing and deduplicating sent tech news articles.
Provides multi-layer deduplication:
  1. Exact URL matching
  2. Canonical/Normalized URL matching (strips UTM tags, tracking params, trailing slashes, www)
  3. Fuzzy Semantic/Title matching across sources and historically (detects when multiple outlets report on the same event)
  4. In-batch cross-publication deduplication
"""

import os
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_PATH = os.path.join(_ROOT_DIR, "data", "sent.db")
# Fallback to root sent.db if data/sent.db doesn't exist yet but root sent.db exists
if not os.path.exists(DEFAULT_DB_PATH) and os.path.exists(os.path.join(_ROOT_DIR, "sent.db")):
    DEFAULT_DB_PATH = os.path.join(_ROOT_DIR, "sent.db")

# Common stopwords to exclude when analyzing title keywords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours",
    "says", "new", "now", "will", "first", "amid", "via", "report", "reports", "today", "here"
}

TRACKING_PARAMS: Set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "guccounter", "guce_referrer", "ncid", "src",
    "spm", "from", "sub_rt", "feed",
}


def normalize_url(url: str) -> str:
    """Canonicalize a URL to strip tracking parameters, fragments, www, and trailing slashes."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path.rstrip("/")

        # Filter out tracking query parameters
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        cleaned_query = [
            (k, v) for k, v in query_pairs if k.lower() not in TRACKING_PARAMS
        ]
        cleaned_query.sort()
        query_str = urlencode(cleaned_query)

        return urlunparse((scheme, netloc, path, "", query_str, ""))
    except Exception:
        return url.strip().rstrip("/")


def stem_word(w: str) -> str:
    """Basic English suffix stemming for plural and tense normalization."""
    w = w.lower()
    for suffix in ("ing", "ies", "es", "s", "ed"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    return w


def extract_keywords(title: str) -> Set[str]:
    """Extract substantive stemmed keywords (content words) from a headline."""
    clean = re.sub(r"[^\w\s]", " ", title.lower())
    words = clean.split()
    return {stem_word(w) for w in words if w not in STOPWORDS and len(w) > 2}


def are_titles_similar(t1: str, t2: str) -> bool:
    """Determine if two titles refer to the same news event."""
    if not t1 or not t2:
        return False

    s1 = t1.strip().lower()
    s2 = t2.strip().lower()

    if s1 == s2:
        return True

    # If numbers or versions explicitly differ (e.g. 'Article 1' vs 'Article 2', 'iPhone 15' vs 'iPhone 16')
    digits1 = re.findall(r"\b\d+\b", s1)
    digits2 = re.findall(r"\b\d+\b", s2)
    if digits1 and digits2 and set(digits1) != set(digits2):
        return False

    k1 = extract_keywords(t1)
    k2 = extract_keywords(t2)
    if not k1 or not k2:
        return False

    # For very short titles (< 3 keywords), require identical keyword sets
    if len(k1) < 3 and len(k2) < 3:
        return k1 == k2

    inter = k1.intersection(k2)
    if len(inter) >= 2:
        # If at least 3 keywords match, or at least one is a substantial entity (>= 5 chars)
        if len(inter) >= 3 or any(len(w) >= 5 for w in inter):
            return True
        if len(inter) / min(len(k1), len(k2)) >= 0.35:
            return True

    # Sequence matcher for longer headlines
    if len(s1) > 20 and len(s2) > 20 and SequenceMatcher(None, s1, s2).ratio() >= 0.65:
        return True

    return False


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the SQLite database schema and migrate columns if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_articles (
                url TEXT PRIMARY KEY,
                normalized_url TEXT,
                title TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Migrate existing table if missing columns
        existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(sent_articles)").fetchall()]
        if "normalized_url" not in existing_cols:
            cursor.execute("ALTER TABLE sent_articles ADD COLUMN normalized_url TEXT")
        if "title" not in existing_cols:
            cursor.execute("ALTER TABLE sent_articles ADD COLUMN title TEXT")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent_normalized ON sent_articles(normalized_url)")
        conn.commit()


def get_sent_records(db_path: str = DEFAULT_DB_PATH) -> List[Tuple[str, str, str]]:
    """Retrieve all previously sent records as (raw_url, normalized_url, title)."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url, normalized_url, title FROM sent_articles")
        rows = cursor.fetchall()
        return [
            (r[0] or "", r[1] or "", r[2] or "")
            for r in rows
        ]


def get_sent_urls(db_path: str = DEFAULT_DB_PATH) -> Set[str]:
    """Retrieve the set of all URLs that have already been sent."""
    records = get_sent_records(db_path)
    sent_urls = set()
    for raw_url, norm_url, _ in records:
        if raw_url:
            sent_urls.add(raw_url)
        if norm_url:
            sent_urls.add(norm_url)
    return sent_urls


def filter_unsent_articles(
    articles: List[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Filter out duplicate articles using URL canonicalization, historical title matching, and in-batch deduplication."""
    records = get_sent_records(db_path)
    sent_raw_urls = {r[0] for r in records if r[0]}
    sent_norm_urls = {r[1] for r in records if r[1]}
    sent_titles = [r[2] for r in records if r[2]]

    fresh: List[Dict[str, Any]] = []
    seen_batch_norm_urls: Set[str] = set()
    seen_batch_titles: List[str] = []

    for article in articles:
        raw_url = article.get("url", "").strip()
        title = article.get("title", "").strip()

        if not raw_url:
            continue

        norm_url = normalize_url(raw_url)

        # 1. Check exact or normalized URL in database
        if raw_url in sent_raw_urls or norm_url in sent_norm_urls:
            continue

        # 2. Check URL in current batch
        if norm_url in seen_batch_norm_urls:
            continue

        # 3. Check title similarity against all previously sent articles
        is_historical_dupe = False
        if title:
            for past_title in sent_titles:
                if are_titles_similar(title, past_title):
                    is_historical_dupe = True
                    break
        if is_historical_dupe:
            continue

        # 4. Check title similarity within the current batch (cross-outlet duplicate)
        is_batch_dupe = False
        if title:
            for b_title in seen_batch_titles:
                if are_titles_similar(title, b_title):
                    is_batch_dupe = True
                    break
        if is_batch_dupe:
            continue

        # Passes all duplicate checks: genuine new story!
        seen_batch_norm_urls.add(norm_url)
        if title:
            seen_batch_titles.append(title)
        fresh.append(article)

    return fresh


def mark_articles_as_sent(
    articles: Iterable[Dict[str, Any]], db_path: str = DEFAULT_DB_PATH
) -> None:
    """Record a collection of articles (with URL, normalized URL, and title) as sent."""
    init_db(db_path)
    rows_to_insert = []
    for a in articles:
        url = a.get("url", "").strip()
        if not url:
            continue
        norm_url = normalize_url(url)
        title = a.get("title", "").strip()
        rows_to_insert.append((url, norm_url, title))

    if not rows_to_insert:
        return

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO sent_articles (url, normalized_url, title)
            VALUES (?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()


def mark_urls_as_sent(
    urls: Iterable[str], db_path: str = DEFAULT_DB_PATH
) -> None:
    """Backward-compatible helper to record URLs as sent."""
    articles = [{"url": u, "title": ""} for u in urls if u]
    mark_articles_as_sent(articles, db_path)
