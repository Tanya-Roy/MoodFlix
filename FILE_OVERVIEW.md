# Project file overview

## App and active modules

- `app.py` — Streamlit interface: name-based local profiles, 10-favorite
  onboarding, voice/text requests, TMDb/Wikipedia poster lookup,
  recommendation cards, and per-movie feedback.
- `profile_store.py` — SQLite persistence for saved profile preferences,
  selected favorites, derived mood intent, movie ratings, relevance scores, and
  watchlist/pass actions.
- `taste_recommender.py` — Content-first ranking, the 10/30/60 release-era
  mix, and K-means profile clustering with same-cluster feedback refinement.
- `content_based_similarity.py` — Loads the MovieLens feature CSV, creates
  sparse TF-IDF movie vectors, and provides the content model used by the app.
- `mood_detector_distilBert.py` — Loads the locally fine-tuned six-emotion
  DistilBERT checkpoint, applies curated mood/context rules, and extracts
  explicit requested/avoided genres separately from emotion.
- `train_distilbert_emotion.py` — Fine-tunes base DistilBERT on the public
  `dair-ai/emotion` dataset, chooses by validation macro-F1, and writes held-out
  test metrics and the local model under `models\distilbert-emotion\`.
- `download_movielens_32m.py` — Downloads and prepares the MovieLens 32M data
  if the cleaned CSVs are missing.
- `collaborative_filtering.py` — Separate sparse MovieLens rating model and
  offline Precision@K, Recall@K, and NDCG@K evaluation command.
- `tests\test_mood_genre_requests.py` — Focused tests for genre-intent phrase
  extraction and negation handling.
- `tests\test_project_basics.py` — Tests local profile/feedback persistence,
  recommendation era quotas, and the low-profile-count clustering fallback.

## Data and local state

- `movielensRatings\ml-32m\cleaned\` — Ratings and movie feature CSVs consumed by
  the recommender. MovieLens 32M ratings run through October 2023.
- `data\moodflix_profiles.sqlite3` — Automatically created local profile and
  feedback database. It is ignored by Git and is not encrypted.
- `50kReviews\`, `imdbMetadata\` — Historical/auxiliary datasets not required
  by the current Streamlit app.

## Setup and documentation

- `requirements.txt` — Python dependencies, including Streamlit, scikit-learn,
  Transformers, PyTorch, certifi, and the Groq SDK.
- `setup_venv.ps1`, `setup_venv.sh` — Virtual-environment setup helpers.
- `README.md` — VS Code setup, API keys, workflow, model limitations, and
  evaluation instructions.
