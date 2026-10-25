import unittest

from mood_detector_distilBert import extract_genre_requests


class GenreRequestTests(unittest.TestCase):
    def test_extracts_multiple_explicit_genres(self):
        preferred, avoided = extract_genre_requests(
            "something funny but with a bit of heart"
        )
        self.assertEqual(preferred, ["Comedy", "Drama"])
        self.assertEqual(avoided, [])

    def test_detects_negated_genre(self):
        preferred, avoided = extract_genre_requests(
            "I do not want a horror movie; give me a comedy"
        )
        self.assertEqual(preferred, ["Comedy"])
        self.assertEqual(avoided, ["Horror"])

    def test_keeps_emotion_and_genre_as_distinct_concepts(self):
        preferred, avoided = extract_genre_requests(
            "I feel sad, but I want an exciting action movie"
        )
        self.assertEqual(preferred, ["Action"])
        self.assertEqual(avoided, [])

    def test_ignores_genre_substrings_inside_other_words(self):
        preferred, avoided = extract_genre_requests(
            "I want a joyful story about community"
        )
        self.assertEqual(preferred, [])
        self.assertEqual(avoided, [])


if __name__ == "__main__":
    unittest.main()
