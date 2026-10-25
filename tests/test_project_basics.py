import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scipy.sparse import csr_matrix

import profile_store
from taste_recommender import balanced_recommendations, peer_feedback_scores


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "profiles.sqlite3"
        self.database_patch = patch.object(
            profile_store, "DATABASE_PATH", self.database_path
        )
        self.database_patch.start()
        profile_store.initialize_store()

    def tearDown(self):
        self.database_patch.stop()
        self.temp_dir.cleanup()

    def test_profile_intent_and_feedback_round_trip(self):
        profile_id = profile_store.save_profile(
            "  Test Profile  ",
            "Malayalam",
            ["Comedy", "Drama"],
            "Warm stories",
            [12, 34],
        )
        profile_store.save_latest_intent(
            profile_id,
            {"mood": "joy", "requested_genres": ["Comedy"]},
        )
        profile_store.save_feedback(
            profile_id,
            56,
            appropriateness=5,
            movie_rating=4,
            action="watchlist",
        )

        profile = profile_store.load_profile("test profile")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "Test Profile")
        self.assertEqual(profile["preferred_language"], "Malayalam")
        self.assertEqual(profile["preferred_genres"], ["Comedy", "Drama"])
        self.assertEqual(profile["other_preferences"], "Warm stories")
        self.assertEqual(profile["liked_movie_ids"], [12, 34])
        self.assertEqual(
            profile["latest_intent"],
            {"mood": "joy", "requested_genres": ["Comedy"]},
        )
        self.assertEqual(
            profile_store.load_feedback(profile_id),
            [
                {
                    "item_id": 56,
                    "appropriateness": 5,
                    "movie_rating": 4,
                    "action": "watchlist",
                }
            ],
        )

    def test_store_initialization_is_idempotent(self):
        profile_store.initialize_store()
        profile_id = profile_store.save_profile("Again", "Any language", [], "", [])
        self.assertIsNotNone(profile_store.load_profile("Again"))
        self.assertGreater(profile_id, 0)


class RecommendationBasicsTests(unittest.TestCase):
    def test_balanced_recommendations_enforce_two_six_twelve_mix(self):
        movies = pd.DataFrame(
            [
                {
                    "item_id": item_id,
                    "release_year": year,
                    "final_score": float(item_id),
                }
                for item_id, year in enumerate(
                    [1990, 1999]
                    + [2000, 2005, 2010, 2012, 2014, 2015]
                    + [
                        2016, 2017, 2018, 2019, 2020, 2021,
                        2022, 2023, 2016, 2017, 2018, 2019,
                    ],
                    start=1,
                )
            ]
        )

        result = balanced_recommendations(movies, count_per_era=1)

        self.assertEqual(len(result), 20)
        self.assertEqual((result["release_year"] < 2000).sum(), 2)
        self.assertEqual(result["release_year"].between(2000, 2015).sum(), 6)
        self.assertEqual((result["release_year"] >= 2016).sum(), 12)

    def test_balancing_reports_when_era_quota_cannot_be_met(self):
        movies = pd.DataFrame(
            {
                "release_year": [1999, 2000, 2001],
                "final_score": [1.0, 0.9, 0.8],
            }
        )
        with self.assertRaisesRegex(ValueError, "release-year distribution"):
            balanced_recommendations(movies, count_per_era=1)

    def test_peer_signal_falls_back_with_fewer_than_three_profiles(self):
        profiles = [
            {"profile_id": 1, "liked_movie_ids": [10], "feedback": []},
            {"profile_id": 2, "liked_movie_ids": [20], "feedback": []},
        ]
        content_model = type(
            "ContentModel",
            (),
            {
                "id_to_idx": {10: 0, 20: 1},
                "tfidf_matrix": csr_matrix([[1.0, 0.0], [0.0, 1.0]]),
            },
        )()
        with patch(
            "taste_recommender.load_all_profiles_for_clustering",
            return_value=profiles,
        ):
            scores, peer_count = peer_feedback_scores(1, content_model)

        self.assertEqual(scores, {})
        self.assertEqual(peer_count, 0)


if __name__ == "__main__":
    unittest.main()
