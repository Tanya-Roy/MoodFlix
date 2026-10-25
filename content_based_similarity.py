"""
Phase 3 - Content-Based Recommender
====================================
Reads items_with_content_features.csv (output of the Phase 2 script),
vectorizes the "combined_features" soup with TF-IDF, and returns the
top-N most similar movies to a given item_id via cosine similarity.
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
ITEMS_PATH = (
    PROJECT_DIR
    / "movielensRatings"
    / "ml-32m"
    / "cleaned"
    / "items_with_content_features.csv"
)


class ContentBasedRecommender:
    def __init__(self, items_path=ITEMS_PATH):
        items_path = Path(items_path)
        if not items_path.is_file():
            raise FileNotFoundError(
                f"MovieLens 32M movie features not found at {items_path}. "
                "Run download_movielens_32m.py first."
            )
        self.items = pd.read_csv(items_path)

        self.items["combined_features"] = self.items["combined_features"].fillna("")
        empty_mask = self.items["combined_features"].str.strip() == ""
        self.n_empty = empty_mask.sum()
        if self.n_empty:
            print(f"[warn] {self.n_empty} movies have an empty feature soup.")

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.items["combined_features"])

        self.id_to_idx = pd.Series(self.items.index, index=self.items["item_id"]).to_dict()

    def get_similar(self, item_id, top_n=10):
        if item_id not in self.id_to_idx:
            raise ValueError(f"item_id {item_id} not found in items table.")

        idx = self.id_to_idx[item_id]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        sim_scores = [(i, s) for i, s in enumerate(sims) if i != idx]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        top = sim_scores[:top_n]

        result = self.items.iloc[[i for i, _ in top]][["item_id", "title", "combined_features"]].copy()
        result["similarity"] = [score for _, score in top]
        return result.reset_index(drop=True)

    def get_by_title(self, query, top_n=10):
        matches = self.items[self.items["clean_title"].str.contains(query, case=False, na=False)]
        if matches.empty:
            raise ValueError(f"No movie title matches '{query}'")
        item_id = matches.iloc[0]["item_id"]
        print(f"Using match: {matches.iloc[0]['title']} (item_id={item_id})")
        return self.get_similar(item_id, top_n=top_n)


if __name__ == "__main__":
    rec = ContentBasedRecommender()
    print(f"Loaded {len(rec.items)} movies, TF-IDF shape: {rec.tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(rec.vectorizer.vocabulary_)}")
    print()

    for query in ["Star Wars", "Toy Story", "Terminator"]:
        try:
            print(f"--- Similar to '{query}' ---")
            results = rec.get_by_title(query, top_n=5)
            print(results[["item_id", "title", "similarity"]].to_string(index=False))
            print()
        except ValueError as e:
            print(e)