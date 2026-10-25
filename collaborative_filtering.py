"""Sparse collaborative filtering for MovieLens ratings using truncated SVD."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "movielensRatings" / "ml-32m" / "cleaned"
RATINGS_PATH = DATA_DIR / "ratings_clean.csv"
ITEMS_PATH = DATA_DIR / "items_clean.csv"


class CollaborativeRecommender:
    """Factorize a sparse user-item rating matrix without materializing predictions."""

    def __init__(self, ratings_path=RATINGS_PATH, n_factors=50):
        ratings_path = Path(ratings_path)
        if not ratings_path.is_file():
            raise FileNotFoundError(
                f"MovieLens 32M ratings not found at {ratings_path}. "
                "Run download_movielens_32m.py first."
            )
        ratings = pd.read_csv(
            ratings_path,
            dtype={
                "user_id": "int32",
                "item_id": "int32",
                "rating": "float32",
                "timestamp": "int32",
            },
        )
        self.n_factors = n_factors
        self._fit(ratings)

    def _fit(self, ratings_df):
        required = {"user_id", "item_id", "rating"}
        missing = required.difference(ratings_df.columns)
        if missing:
            raise ValueError(f"Ratings data is missing required columns: {sorted(missing)}")

        self.user_ids = np.sort(ratings_df["user_id"].unique())
        self.item_ids = np.sort(ratings_df["item_id"].unique())
        self.user_id_to_row = {user_id: row for row, user_id in enumerate(self.user_ids)}
        self.item_id_to_col = {item_id: col for col, item_id in enumerate(self.item_ids)}

        rows = ratings_df["user_id"].map(self.user_id_to_row).to_numpy()
        cols = ratings_df["item_id"].map(self.item_id_to_col).to_numpy()
        values = ratings_df["rating"].to_numpy(dtype=np.float32)
        self.matrix = coo_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        ).tocsr()

        rated_counts = np.diff(self.matrix.indptr)
        rating_sums = np.asarray(self.matrix.sum(axis=1)).ravel()
        self.user_means = np.zeros(len(self.user_ids), dtype=np.float32)
        np.divide(rating_sums, rated_counts, out=self.user_means, where=rated_counts > 0)

        centered = self.matrix.copy()
        row_indices = np.repeat(np.arange(centered.shape[0]), np.diff(centered.indptr))
        centered.data -= self.user_means[row_indices]

        max_factors = min(centered.shape) - 1
        if max_factors < 1:
            raise ValueError("Not enough users and movies to fit the collaborative model.")
        factors = min(self.n_factors, max_factors)
        user_factors, singular_values, item_factors = svds(centered, k=factors)
        order = np.argsort(singular_values)[::-1]
        self.user_factors = user_factors[:, order].astype(np.float32, copy=False)
        self.singular_values = singular_values[order].astype(np.float32, copy=False)
        self.item_factors = item_factors[order, :].astype(np.float32, copy=False)

    def predict_for_user(self, user_id):
        """Return scores for every item for one user using only a small dense vector."""
        row = self.user_id_to_row.get(user_id)
        if row is None:
            raise ValueError(f"user_id {user_id} not found in ratings.")
        scores = (self.user_factors[row] * self.singular_values) @ self.item_factors
        return np.clip(scores + self.user_means[row], 1, 5)

    def predict(self, user_id, item_id):
        row = self.user_id_to_row.get(user_id)
        col = self.item_id_to_col.get(item_id)
        if row is None or col is None:
            return None
        score = (
            (self.user_factors[row] * self.singular_values)
            @ self.item_factors[:, col]
        ) + self.user_means[row]
        return float(np.clip(score, 1, 5))

    def recommend_for_user(self, user_id, top_n=10, items_meta_path=ITEMS_PATH):
        if user_id not in self.user_id_to_row:
            raise ValueError(f"user_id {user_id} not found (cold-start user).")

        scores = self.predict_for_user(user_id)
        row = self.user_id_to_row[user_id]
        rated_cols = self.matrix.indices[
            self.matrix.indptr[row] : self.matrix.indptr[row + 1]
        ]
        already_rated = {self.item_ids[col] for col in rated_cols}
        candidates = [
            (item_id, scores[col])
            for col, item_id in enumerate(self.item_ids)
            if item_id not in already_rated
        ]
        candidates.sort(key=lambda entry: entry[1], reverse=True)
        result = pd.DataFrame(
            candidates[:top_n], columns=["item_id", "predicted_rating"]
        )

        metadata_path = Path(items_meta_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Movie metadata not found at {metadata_path}.")
        items = pd.read_csv(metadata_path)[["item_id", "title"]]
        return (
            result.merge(items, on="item_id")
            .sort_values("predicted_rating", ascending=False)
            .reset_index(drop=True)
        )


def evaluate_rmse(
    ratings_path=RATINGS_PATH,
    n_factors=50,
    test_size=0.2,
    random_state=42,
):
    """Train on a random ratings split and report RMSE for known users/items."""
    ratings = pd.read_csv(
        ratings_path,
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int32",
        },
    )
    train, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    model = CollaborativeRecommender.__new__(CollaborativeRecommender)
    model.n_factors = n_factors
    model._fit(train)

    known = test[
        test["user_id"].isin(model.user_id_to_row)
        & test["item_id"].isin(model.item_id_to_col)
    ]
    predictions = [
        model.predict(row.user_id, row.item_id)
        for row in known.itertuples(index=False)
    ]
    if not predictions:
        raise ValueError("The held-out split contains no known user/movie pairs.")
    rmse = np.sqrt(mean_squared_error(known["rating"], predictions))
    print(f"RMSE on {len(predictions)} held-out ratings: {rmse:.4f}")
    return rmse


def evaluate_ranking_metrics(
    ratings_path=RATINGS_PATH,
    k=10,
    max_users=1000,
    min_ratings_per_user=10,
    random_state=42,
):
    """Evaluate top-k recommendations against each user's latest held-out ratings."""
    if k < 1 or max_users < 1 or min_ratings_per_user < 2:
        raise ValueError(
            "k and max_users must be positive; min_ratings_per_user must be at least 2."
        )
    ratings = pd.read_csv(
        ratings_path,
        dtype={
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int32",
        },
    )
    eligible_counts = ratings.groupby("user_id").size()
    eligible_users = eligible_counts[eligible_counts >= min_ratings_per_user].index
    if len(eligible_users) == 0:
        raise ValueError(
            f"No users have at least {min_ratings_per_user} ratings for evaluation."
        )
    sampled_users = pd.Series(eligible_users).sample(
        n=min(max_users, len(eligible_users)),
        random_state=random_state,
    )
    evaluation_ratings = ratings[ratings["user_id"].isin(sampled_users)].sort_values(
        ["user_id", "timestamp"], kind="stable"
    )

    training_parts = []
    heldout_parts = []
    for _, user_ratings in evaluation_ratings.groupby("user_id", sort=False):
        split_at = max(int(len(user_ratings) * 0.8), 1)
        training_parts.append(user_ratings.iloc[:split_at])
        heldout_parts.append(user_ratings.iloc[split_at:])
    train = pd.concat(training_parts, ignore_index=True)
    heldout = pd.concat(heldout_parts, ignore_index=True)

    model = CollaborativeRecommender.__new__(CollaborativeRecommender)
    model.n_factors = 50
    model._fit(train)

    precision_values = []
    recall_values = []
    ndcg_values = []
    for user_id, user_test in heldout.groupby("user_id"):
        relevant_items = set(user_test.loc[user_test["rating"] >= 4, "item_id"])
        if not relevant_items or user_id not in model.user_id_to_row:
            continue

        user_row = model.user_id_to_row[user_id]
        scores = model.predict_for_user(user_id)
        rated_cols = model.matrix.indices[
            model.matrix.indptr[user_row] : model.matrix.indptr[user_row + 1]
        ]
        scores[rated_cols] = -np.inf
        candidate_count = min(k, len(scores) - len(rated_cols))
        if candidate_count <= 0:
            continue
        top_cols = np.argpartition(scores, -candidate_count)[-candidate_count:]
        top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
        recommended_items = model.item_ids[top_cols]
        hits = np.asarray(
            [1 if item_id in relevant_items else 0 for item_id in recommended_items]
        )

        precision_values.append(float(hits.sum()) / k)
        recall_values.append(float(hits.sum()) / len(relevant_items))
        discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float(np.sum(hits * discounts))
        ideal_hits = min(len(relevant_items), k)
        ideal_dcg = float(np.sum(1 / np.log2(np.arange(2, ideal_hits + 2))))
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg else 0.0)

    if not precision_values:
        raise ValueError(
            "No users had relevant held-out ratings that could be evaluated."
        )
    metrics = {
        f"precision@{k}": float(np.mean(precision_values)),
        f"recall@{k}": float(np.mean(recall_values)),
        f"ndcg@{k}": float(np.mean(ndcg_values)),
        "evaluated_users": len(precision_values),
    }
    print("Ranking metrics (temporal 80/20 split, sampled users):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate-ranking",
        action="store_true",
        help="Report Precision@K, Recall@K, and NDCG@K on a sampled user set.",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1000)
    args = parser.parse_args()

    if args.k < 1 or args.max_users < 1:
        parser.error("--k and --max-users must be positive integers.")
    if args.evaluate_ranking:
        evaluate_ranking_metrics(k=args.k, max_users=args.max_users)
    else:
        recommender = CollaborativeRecommender(n_factors=50)
        print(f"Sparse rating matrix shape (users x movies): {recommender.matrix.shape}")
        sample_user = recommender.user_ids[0].item()
        print(recommender.recommend_for_user(sample_user, top_n=10).to_string(index=False))
