"""Content-first ranking with local, cluster-based collaborative feedback."""

import ast
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from profile_store import load_all_profiles_for_clustering


ERA_BUCKETS = (
    ("before 2000", 0, 1999, 2),
    ("2000–2015", 2000, 2015, 6),
    ("2016 and newer", 2016, None, 12),
)


def parse_genres(value) -> list[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return result if isinstance(result, list) else []


def _ids_to_vector(item_ids, content_model):
    rows = [
        content_model.id_to_idx[item_id]
        for item_id in item_ids
        if item_id in content_model.id_to_idx
    ]
    if not rows:
        return None
    return np.asarray(
        content_model.tfidf_matrix[rows].mean(axis=0)
    ).reshape(1, -1)


def peer_feedback_scores(profile_id: int, content_model) -> tuple[dict, int]:
    """Cluster saved profiles by favorite-movie content, then aggregate peer feedback."""
    profiles = load_all_profiles_for_clustering()
    usable_profiles = []
    vectors = []
    for profile in profiles:
        positive_feedback = [
            entry["item_id"]
            for entry in profile["feedback"]
            if entry["action"] == "watchlist"
            or (entry["movie_rating"] is not None and entry["movie_rating"] >= 4)
            or (
                entry["movie_rating"] is None
                and entry["appropriateness"] >= 4
            )
        ]
        profile_vector = _ids_to_vector(
            list(dict.fromkeys(profile["liked_movie_ids"] + positive_feedback)),
            content_model,
        )
        if profile_vector is not None:
            usable_profiles.append(profile)
            vectors.append(np.asarray(profile_vector).ravel())

    if len(usable_profiles) < 3:
        return {}, 0

    n_clusters = min(5, max(2, round(math.sqrt(len(usable_profiles)))))
    n_clusters = min(n_clusters, len(usable_profiles) - 1)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
        np.vstack(vectors)
    )
    current_index = next(
        (
            index
            for index, profile in enumerate(usable_profiles)
            if profile["profile_id"] == profile_id
        ),
        None,
    )
    if current_index is None:
        return {}, 0

    peer_profiles = [
        profile
        for index, profile in enumerate(usable_profiles)
        if labels[index] == labels[current_index] and index != current_index
    ]
    movie_scores: dict[int, list[float]] = {}
    for peer in peer_profiles:
        for item_id in peer["liked_movie_ids"]:
            movie_scores.setdefault(item_id, []).append(1.0)
        for entry in peer["feedback"]:
            relevance = (entry["appropriateness"] - 1) / 4
            if entry["movie_rating"] is not None:
                movie_quality = (entry["movie_rating"] - 1) / 4
                score = 0.6 * relevance + 0.4 * movie_quality
            else:
                score = relevance
            if entry["action"] == "watchlist":
                score = max(score, 0.9)
            elif entry["action"] == "pass":
                score = min(score, 0.15)
            movie_scores.setdefault(entry["item_id"], []).append(score)

    return {
        item_id: float(np.mean(scores))
        for item_id, scores in movie_scores.items()
    }, len(peer_profiles)


def _era_mask(years: pd.Series, first: int, last: int | None) -> pd.Series:
    numeric = pd.to_numeric(years, errors="coerce")
    if last is None:
        return numeric >= first
    return numeric.between(first, last, inclusive="both")


def balanced_recommendations(
    movies: pd.DataFrame,
    score_column: str = "final_score",
    count_per_era: int = 20,
) -> pd.DataFrame:
    """Return 10/30/60 era proportions; count_per_era=1 returns 2/6/12."""
    selected = []
    missing = []
    for label, first_year, last_year, quota in ERA_BUCKETS:
        bucket = movies.loc[
            _era_mask(movies["release_year"], first_year, last_year)
        ].sort_values(score_column, ascending=False)
        take = count_per_era * quota
        selected.append(bucket.head(take))
        if len(bucket) < take:
            missing.append(f"{label}: {len(bucket)}/{take}")
    if missing:
        raise ValueError(
            "Cannot meet the requested release-year distribution "
            f"({'; '.join(missing)}). Loosen genre or language preferences."
        )
    return (
        pd.concat(selected)
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )


def rank_recommendations(
    content_model,
    profile: dict,
    mood_profile: dict,
    request_text: str,
    preferred_language: str,
    language_codes: dict[str, str],
    tmdb_info_lookup=None,
) -> tuple[pd.DataFrame, int, bool]:
    """Rank by liked-film similarity first and same-cluster feedback second."""
    items = content_model.items.copy()
    items["release_year"] = pd.to_numeric(items["release_year"], errors="coerce")
    liked_ids = list(dict.fromkeys(profile["liked_movie_ids"]))
    user_ratings = {
        entry["item_id"]: entry
        for entry in profile.get("feedback", [])
    }
    seed_ids = liked_ids + [
        item_id
        for item_id, feedback in user_ratings.items()
        if feedback["action"] == "watchlist"
        or (feedback["movie_rating"] is not None and feedback["movie_rating"] >= 4)
        or (
            feedback["movie_rating"] is None
            and feedback["appropriateness"] >= 4
        )
    ]
    seed_vector = _ids_to_vector(list(dict.fromkeys(seed_ids)), content_model)
    if seed_vector is None:
        raise ValueError("This profile has no usable liked movies to build recommendations.")

    content_scores = cosine_similarity(
        content_model.tfidf_matrix, seed_vector
    ).ravel()
    items["content_score"] = content_scores
    items["final_score"] = items["content_score"]

    explicit_genres = set(mood_profile.get("requested_genres", []))
    explicit_avoid_genres = set(mood_profile.get("requested_avoid_genres", []))
    preferred_genres = set(profile["preferred_genres"])
    if explicit_genres:
        preferred_genres.update(explicit_genres)
        mood_genres = set()
    else:
        mood_genres = set(mood_profile.get("preferred_genres", []))
    avoid_genres = (
        set(
            mood_profile.get(
                "mood_avoid_genres",
                mood_profile.get("avoid_genres", []),
            )
        )
        - explicit_genres
    ) | explicit_avoid_genres
    movie_genres = items["genres_list"].map(parse_genres)
    if preferred_genres:
        items["final_score"] += movie_genres.map(
            lambda genres: (
                0.30 * len(explicit_genres.intersection(genres))
                + 0.12 * len(
                    (preferred_genres - explicit_genres).intersection(genres)
                )
            )
        )
    if mood_genres:
        items["final_score"] += movie_genres.map(
            lambda genres: 0.08 if mood_genres.intersection(genres) else 0.0
        )
    if avoid_genres:
        items["final_score"] -= movie_genres.map(
            lambda genres: 0.20 if avoid_genres.intersection(genres) else 0.0
        )

    if request_text.strip():
        query_vector = content_model.vectorizer.transform([request_text.strip()])
        items["final_score"] += 0.2 * cosine_similarity(
            content_model.tfidf_matrix, query_vector
        ).ravel()

    negative_item_ids = [
        item_id
        for item_id, feedback in user_ratings.items()
        if feedback["appropriateness"] <= 2
        or (feedback["movie_rating"] is not None and feedback["movie_rating"] <= 2)
        or feedback["action"] == "pass"
    ]
    negative_rows = [
        content_model.id_to_idx[item_id]
        for item_id in negative_item_ids
        if item_id in content_model.id_to_idx
    ]
    if negative_rows:
        disliked_vector = np.asarray(
            content_model.tfidf_matrix[negative_rows].mean(axis=0)
        ).reshape(1, -1)
        items["final_score"] -= 0.2 * cosine_similarity(
            content_model.tfidf_matrix, disliked_vector
        ).ravel()

    if liked_ids:
        items = items[~items["item_id"].isin(liked_ids)]
    if user_ratings:
        items = items[~items["item_id"].isin(user_ratings)]
    items = items[items["tconst"].notna()]

    selected_language = mood_profile.get("requested_language") or preferred_language
    language_code = language_codes.get(selected_language)
    peer_scores, peer_count = peer_feedback_scores(
        profile["profile_id"], content_model
    )

    initial_shortlist = balanced_recommendations(
        items.sort_values("final_score", ascending=False),
        count_per_era=3,
    )
    language_filtered = False
    if language_code and tmdb_info_lookup is not None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            movie_infos = list(executor.map(tmdb_info_lookup, initial_shortlist["tconst"]))
        language_matches = [
            movie_info is not None
            and movie_info.get("original_language") == language_code
            for movie_info in movie_infos
        ]
        language_candidates = initial_shortlist.loc[language_matches].copy()
        try:
            initial_shortlist = balanced_recommendations(
                language_candidates.sort_values("final_score", ascending=False),
                count_per_era=1,
            )
            language_filtered = True
        except ValueError:
            pass

    if peer_scores:
        peer_values = initial_shortlist["item_id"].map(peer_scores)
        known_peer = peer_values.notna()
        initial_shortlist.loc[known_peer, "final_score"] = (
            0.8 * initial_shortlist.loc[known_peer, "final_score"]
            + 0.2 * peer_values[known_peer]
        )

    recommendations = balanced_recommendations(
        initial_shortlist.sort_values("final_score", ascending=False),
        count_per_era=1,
    )
    return recommendations.head(20).reset_index(drop=True), peer_count, language_filtered
