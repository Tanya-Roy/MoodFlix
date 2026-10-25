"""Local SQLite storage for named profiles, preferences, and movie feedback."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_DIR / "data" / "moodflix_profiles.sqlite3"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_store() -> None:
    """Create local tables if absent; profile records remain on this machine."""
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                preferred_language TEXT NOT NULL DEFAULT 'Any language',
                preferred_genres_json TEXT NOT NULL DEFAULT '[]',
                other_preferences TEXT NOT NULL DEFAULT '',
                liked_movie_ids_json TEXT NOT NULL DEFAULT '[]',
                latest_intent_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id),
                item_id INTEGER NOT NULL,
                appropriateness INTEGER NOT NULL CHECK (appropriateness BETWEEN 1 AND 5),
                movie_rating INTEGER CHECK (movie_rating BETWEEN 1 AND 5),
                action TEXT NOT NULL CHECK (action IN ('watchlist', 'pass', 'neutral')),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, item_id)
            );
            """
        )


def list_profiles() -> list[str]:
    with _connect() as connection:
        rows = connection.execute("SELECT name FROM profiles ORDER BY name COLLATE NOCASE")
        return [row["name"] for row in rows]


def load_profile(name: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM profiles WHERE name = ? COLLATE NOCASE",
            (name.strip(),),
        ).fetchone()
    if row is None:
        return None

    profile = dict(row)
    profile["preferred_genres"] = json.loads(profile.pop("preferred_genres_json"))
    profile["liked_movie_ids"] = json.loads(profile.pop("liked_movie_ids_json"))
    profile["latest_intent"] = json.loads(profile.pop("latest_intent_json"))
    return profile


def save_profile(
    name: str,
    preferred_language: str,
    preferred_genres: list[str],
    other_preferences: str,
    liked_movie_ids: list[int],
) -> int:
    """Insert/update preferences for a profile name and return its stable ID."""
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Enter a profile name.")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO profiles (
                name, preferred_language, preferred_genres_json,
                other_preferences, liked_movie_ids_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                preferred_language = excluded.preferred_language,
                preferred_genres_json = excluded.preferred_genres_json,
                other_preferences = excluded.other_preferences,
                liked_movie_ids_json = excluded.liked_movie_ids_json,
                updated_at = excluded.updated_at
            """,
            (
                normalized_name,
                preferred_language,
                json.dumps(preferred_genres),
                other_preferences.strip(),
                json.dumps([int(item_id) for item_id in liked_movie_ids]),
                now,
                now,
            ),
        )
        row = connection.execute(
            "SELECT profile_id FROM profiles WHERE name = ? COLLATE NOCASE",
            (normalized_name,),
        ).fetchone()
    return int(row["profile_id"])


def save_latest_intent(profile_id: int, intent: dict) -> None:
    """Persist only structured intent; raw mood/voice transcripts are not saved."""
    with _connect() as connection:
        connection.execute(
            """
            UPDATE profiles
            SET latest_intent_json = ?, updated_at = ?
            WHERE profile_id = ?
            """,
            (
                json.dumps(intent, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                profile_id,
            ),
        )


def save_feedback(
    profile_id: int,
    item_id: int,
    appropriateness: int,
    movie_rating: int | None,
    action: str,
) -> None:
    if not 1 <= appropriateness <= 5:
        raise ValueError("Appropriateness must be scored from 1 to 5.")
    if movie_rating is not None and not 1 <= movie_rating <= 5:
        raise ValueError("Movie rating must be from 1 to 5.")
    if action not in {"watchlist", "pass", "neutral"}:
        raise ValueError(f"Unsupported movie feedback action: {action}")

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                profile_id, item_id, appropriateness, movie_rating, action, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, item_id) DO UPDATE SET
                appropriateness = excluded.appropriateness,
                movie_rating = excluded.movie_rating,
                action = excluded.action,
                updated_at = excluded.updated_at
            """,
            (
                profile_id,
                int(item_id),
                appropriateness,
                movie_rating,
                action,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def load_feedback(profile_id: int) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT item_id, appropriateness, movie_rating, action
            FROM feedback
            WHERE profile_id = ?
            """,
            (profile_id,),
        )
        return [dict(row) for row in rows]


def load_all_profiles_for_clustering() -> list[dict]:
    with _connect() as connection:
        profiles = connection.execute(
            """
            SELECT profile_id, liked_movie_ids_json
            FROM profiles
            ORDER BY profile_id
            """
        ).fetchall()
        feedback = connection.execute(
            """
            SELECT profile_id, item_id, appropriateness, movie_rating, action
            FROM feedback
            """
        ).fetchall()

    feedback_by_profile: dict[int, list[dict]] = {}
    for row in feedback:
        feedback_by_profile.setdefault(row["profile_id"], []).append(dict(row))
    return [
        {
            "profile_id": row["profile_id"],
            "liked_movie_ids": json.loads(row["liked_movie_ids_json"]),
            "feedback": feedback_by_profile.get(row["profile_id"], []),
        }
        for row in profiles
    ]
