# MoodFlix Project Log

This is the running engineering and product log for MoodFlix. It records the
requirements that shaped the project, approaches that were tried or replaced,
implementation decisions, measured results, and work still outstanding. New
work should be appended as dated entries; do not replace earlier decisions
without recording why they changed.

## How to maintain this log

For each significant future change, add an entry near the top of **Change
history** with:

1. Date and short change title.
2. User need or problem being addressed.
3. Approach selected and important alternatives considered.
4. Files/data affected and observable behavior changes.
5. Validation actually run, including exact commands and outcomes.
6. Measured results, including dataset/split and metric definitions when
   reporting ML metrics.
7. Remaining limitations, risks, and the next step.

Never estimate or invent an accuracy, F1 score, training duration, or test
outcome. If a run did not finish or its report was not saved, mark the result
as pending and update it when evidence is available.

---

## Resume-ready technical inventory

Use this section as a concise index of the technologies and methods implemented
in MoodFlix. The bullets below describe separate project components; the
MovieLens rating-factorization model is an offline recommender and is not the
same model as the profile-based Streamlit app.

### Technologies and methods implemented

- **Languages and application:** Python; Streamlit interactive web application;
  pandas and NumPy for tabular data and numerical processing.
- **Data ingestion and preparation:** MovieLens 32M download and CSV cleaning;
  movie metadata feature construction; sparse SciPy matrices for large-scale
  user-item ratings and movie features.
- **Content-based recommendation:** TF-IDF vectorization of movie feature text;
  sparse feature representations; cosine similarity; ranking over profile
  favorites, current-request features, genre preferences, and saved feedback;
  release-era quotas and language filtering.
- **Collaborative recommendation:** Hybrid ranking in the profile app, combining
  content scores with a secondary peer-feedback signal; taste-vector clustering
  with scikit-learn K-means and a content-only fallback when there are too few
  usable local profiles. A separate MovieLens model uses sparse user-item
  rating matrices and truncated singular value decomposition (SVD).
- **Natural language processing and deep learning:** Hugging Face
  Transformers, Datasets, and PyTorch; fine-tuning DistilBERT for six-class
  emotion classification; curated context rules; separate rule-based
  extraction of explicitly requested or avoided movie genres, including basic
  negation handling.
- **Model training and evaluation:** Train/validation/test workflow;
  validation macro-F1 checkpoint selection; early stopping; class-weighted
  cross-entropy; AdamW optimization; linear learning-rate warmup and decay;
  gradient clipping; optional CUDA mixed precision; accuracy, macro/weighted
  F1, per-class metrics, confusion matrix, and majority-class baseline.
- **Speech and external services:** Streamlit audio input; Groq-hosted Whisper
  speech-to-text transcription; TMDb API for poster and language metadata;
  English Wikipedia API thumbnail fallback; runtime API-key configuration and
  `certifi` TLS certificate roots.
- **Persistence and user feedback:** Local SQLite profile storage; saved
  preferences and favorites; structured mood/genre intent; recommendation
  relevance scores, optional movie ratings, and pass/watchlist actions;
  database constraints for valid feedback values.
- **Evaluation and testing:** MovieLens offline ranking evaluation with
  Precision@K, Recall@K, and NDCG@K; focused Python unit tests for genre intent,
  persistence, recommendation quotas, and sparse-clustering fallback.

### Resume-ready project bullets

- Built **MoodFlix**, a Python/Streamlit personalized movie recommender that
  combines sparse TF-IDF content ranking, profile feedback, release-era
  constraints, and K-means-based peer refinement, with a content-only fallback
  for sparse profile data.
- Fine-tuned a six-class DistilBERT emotion classifier with Hugging Face and
  PyTorch, selecting checkpoints by validation macro-F1; the recorded held-out
  2,000-example public test split scored **92.30% accuracy** and **0.8806
  macro-F1**. These are emotion-dataset metrics, not end-to-end recommendation
  metrics.
- Implemented rule-based extraction of explicit and negated genre intent,
  editable Whisper voice transcription through Groq, TMDb/Wikipedia poster
  retrieval, and local SQLite persistence for profiles and recommendation
  feedback.
- Built a separate sparse MovieLens collaborative-filtering model using
  truncated SVD and offline ranking evaluation with Precision@K, Recall@K, and
  NDCG@K.

---

## Product goal and final intended workflow

MoodFlix is a Streamlit movie recommender intended to:

1. Let someone create a local profile by name, without a password.
2. Save their preferred language, genres, optional taste notes, and ten
   selected favorite movies.
3. Offer a current-request box that accepts typed text or a recorded voice
   request.
4. Use Whisper transcription through Groq for recorded audio, while keeping the
   transcription editable before recommendation generation.
5. Use DistilBERT emotion classification plus a small set of explicit context
   rules to interpret a request.
6. Rank movies content-first from profile favorites, stored feedback, genre
   choices, and the current request.
7. Present poster-backed recommendations and collect a separate
   recommendation-appropriateness score, optional movie rating, and pass /
   watchlist choice.
8. Persist profile and feedback information locally and use similar local
   profiles as a collaborative signal once enough profiles exist.

The current implementation is local-profile based; it is not a multi-user
hosted identity service. The local SQLite file is not encrypted. Posters are
retrieved from TMDb when available or from Wikipedia as a no-key fallback;
language metadata comes from TMDb. Voice transcription in the current version
uses Groq's online service; audio is not deliberately stored by MoodFlix.

---

## Decisions and approaches

### Movie catalog and release-year diversity

- The project moved to MovieLens 32M rather than relying on the earlier smaller
  dataset. Its ratings run through October 2023; it is **not** a dataset current
  through 2026.
- The requested sampling target was interpreted as 10% before 2000, 30% from
  2000 through 2015, and 60% from 2016 onward. For a set of 20, that is 2, 6,
  and 12 titles respectively.
- MovieLens provides ratings and identifiers, not poster images. IMDb IDs from
  the prepared catalog are used to find poster and language metadata through
  TMDb.
- “Random” onboarding choices are seeded per new-profile setup and selected
  across the three release eras. The selector attempts to reduce genre overlap
  among the twenty choices. This is a diversity heuristic, not a guarantee
  that every title is semantically unrelated.
- Recommendation ranking also enforces the 2/6/12 release-year proportions.
  Language filtering is optional and depends on TMDb metadata. If filtering
  cannot supply all three era quotas, recommendations fall back to the other
  preferences rather than showing a partial, misleadingly balanced set.

### Recommendation model

- An earlier plan used MovieLens user IDs and a hybrid MovieLens collaborative
  model. That did not fit the later request for name-based local profiles and
  can be costly when trying to score the entire catalog.
- The current app computes content similarities over the sparse movie TF-IDF
  matrix. Favorite movie vectors form a user taste representation; genre
  preferences and the current text request provide additional ranking signals.
- Saved positive/negative feedback influences subsequent content scores.
- Collaborative refinement is based on local profile taste vectors and saved
  feedback, grouped by deterministic K-means. It is deliberately a secondary
  signal; when there are too few usable profiles, content-based ranking is used
  alone.
- This local-profile collaborative component is distinct from the separate
  MovieLens rating-matrix factorization and its offline evaluation script.
- These methods produce ranking signals, not calibrated probabilities of
  liking a movie.

### Mood model and training strategy

- The product keeps DistilBERT as requested, but the fine-tuning script starts
  from the general-purpose `distilbert-base-uncased` checkpoint. This avoids
  treating an already emotion-fine-tuned model as the supposedly untouched
  starting point for an independent evaluation.
- The public Hugging Face dataset selected is `dair-ai/emotion`, configuration
  `split`. It contains English short texts labelled as sadness, joy, love,
  anger, fear, or surprise, with 16,000 train, 2,000 validation, and 2,000 test
  examples according to its dataset card.
- The train split updates DistilBERT weights. Validation macro-F1 chooses the
  saved checkpoint and supports early stopping. The test split is reserved for
  final evaluation after checkpoint selection.
- The training script calculates accuracy, macro-F1, weighted F1, loss,
  per-class precision/recall/F1, confusion matrix, and a majority-class
  baseline. Results are meant to be written to
  `models\distilbert-emotion\evaluation.json`.
- The app loads the fine-tuned checkpoint locally. It intentionally does not
  silently fall back to a different public emotion model if the fine-tuned
  artifact is missing.
- Dataset fine-tuning teaches broad emotion classification patterns. It does
  not teach the model MoodFlix-specific movie preferences, identify a movie
  genre directly, guarantee that a specific sentence is interpreted
  correctly, or provide medical/mental-health interpretation.
- Genre translation remains a separate set of curated rules. For example,
  breakup phrases currently add a rule to avoid Romance; that behavior is
  hand-authored and is not learned by DistilBERT. Phrase detection for a
  language request (such as “Malayalam”) is also separate from emotion
  classification.
- Explicit movie-genre intent is extracted separately from emotion. The six
  emotion labels are not expanded to include genres: a request for something
  funny is a request for Comedy, not an emotion called “comedy.” The app gives
  explicit requested genres more ranking weight than emotion-inferred genres.
- The project currently has no representative, user-consented, human-labeled
  MoodFlix mood dataset. Movie satisfaction ratings are not valid emotion
  labels and must not be recycled as such.

### Voice input and external services

- Whisper is used for speech-to-text, not mood classification.
- The app calls Groq's hosted Whisper model (`whisper-large-v3-turbo`) using a
  runtime key. Audio is sent to the provider for transcription; MoodFlix stores
  neither the audio nor the raw transcript as part of its profile persistence.
- TMDb is the preferred source for movie posters and original-language
  metadata. If TMDb is not configured, unavailable, or has no poster for a
  title, the app tries an English Wikipedia page thumbnail without a key.
  Wikipedia may return no image or an image that is not the official poster.
  Groq is not a movie-poster generation or movie metadata service.
- API keys are supplied at runtime or through environment variables; they must
  not be hard-coded into source or committed.
- Network/TLS errors have occurred in this environment in earlier app use.
  Poster lookup reports service errors and continues with the other source or
  a text-only card; language filtering cannot be trusted without TMDb metadata.

---

## Change history

### 2026-10-25 — Prepare project for GitHub

**Need:** Prepare MoodFlix for its existing GitHub repository without
publishing local datasets, trained model artifacts, profile data, or secrets.

**Implementation:** Expanded `.gitignore` to exclude the local data/model and
dataset directories, SQLite/database files, and dotenv files. The target
`Tanya-Roy/MoodFlix` repository was confirmed to exist but is private and empty.

**Validation and status:** Checked the project documentation and source files
for credential-like strings; only API-key placeholders and variable names were
found. No files have been pushed. Initializing Git and pushing code, or changing
the repository's visibility, is pending because this environment has no
available PowerShell/Git command runner and the GitHub browser is signed out.

### 2026-10-22 — Add Docker run configuration

**Need:** Provide a repeatable containerized way to build and run the Streamlit
app without packaging the large local dataset, model, or profile database into
the image.

**Implementation:** Added a Python 3.11 slim image definition and Docker Compose
service. Compose bind-mounts the prepared MovieLens files and fine-tuned model
read-only, while keeping profile SQLite data in the project `data` folder.
TMDb and Groq keys are passed as runtime environment variables. Added a
`.dockerignore` to keep local environments, datasets, model artifacts, and
secrets out of the build context, and documented build/run instructions in
`README.md`.

**Validation:** Docker image build and container startup are pending; the
available command runner did not provide a working PowerShell executable.

### 2026-10-13 — Add tests for core app data and ranking behavior

**Need:** Continue the open validation work by adding deterministic checks for
local profile persistence, recommendation release-year quotas, and the
collaborative fallback used before enough local profiles exist.

**Implementation:** Added `tests\test_project_basics.py`. It uses a temporary
SQLite database to check profile, structured-intent, and feedback persistence;
checks the 2/6/12 release-year distribution and its insufficient-catalog error;
and verifies that peer scoring returns an empty signal when fewer than three
profiles are available. The existing genre-intent tests remain part of the
same `unittest` discovery command. Updated `FILE_OVERVIEW.md` and the open-work
list accordingly.

**Validation:** Attempted `python -m unittest discover -s tests` and
`python -m py_compile profile_store.py taste_recommender.py
tests\test_project_basics.py tests\test_mood_genre_requests.py`; the assistant
environment could not launch either command because PowerShell is unavailable.
The tests have not been executed here. Run both commands from the project
directory in the user's activated environment; the Streamlit and optional
service integration checks remain open.

**User-run test follow-up:** The user ran discovery and reported 9 tests with
3 errors. Two profile-store tests failed while deleting their temporary
databases because SQLite connections were not explicitly closed: Python's
SQLite connection context manager handles transactions but does not close the
connection. Updated `_connect()` to close connections in a `finally` block.
The peer-fallback test also passed `None` instead of the content model required
to build profile vectors; it now supplies a two-row in-memory sparse fixture.
The user subsequently confirmed “result ok” after rerunning the commands; the
focused test and compilation checks are therefore recorded as passing (9 tests
reported in the prior run). Detailed output from the successful retry was not
provided.

### 2026-10-13 — Add Wikipedia poster fallback

**User need:** TMDb key setup was not working for the user, and they asked to
use Wikipedia for movie images instead.

**Approach:** Keep TMDb as the preferred source when configured, and query the
English Wikipedia API for a page thumbnail when TMDb is unavailable or has no
poster. Wikipedia lookup needs no API key. Groq remains unrelated and is still
used only for voice transcription.

**Implementation:** `app.py` now searches Wikipedia using the movie title,
release year, and “film,” downloads the first available page thumbnail, caches
lookup results for a day, and falls back to title-only cards when no image is
found. TMDb lookup errors no longer prevent attempting Wikipedia. Sidebar
guidance and retry controls include both sources. `README.md` and
`FILE_OVERVIEW.md` document the behavior and limitations.

**Limitations:** Wikipedia may have no relevant thumbnail, or the thumbnail may
not be the official poster. Images and language metadata remain external;
language filtering still needs TMDb. No runtime test or live API lookup was
performed in the assistant environment.

### 2026-10-13 — Separate explicit genre requests from emotion

**Observed issue:** The fine-tuned model classified “something funny but with a
bit of heart” as `surprise`, after which the old emotion-to-genre mapping
recommended Mystery, Sci-Fi, and Fantasy. Comedy is not one of DistilBERT's six
emotion labels, and the ranking path had not separately retained the user's
explicit genre words.

**Decision:** Keep the six trained emotion labels and add a lightweight
genre-intent extractor, rather than incorrectly adding Comedy as an emotion or
expanding the classifier without a labeled emotion dataset. This preserves the
meaning of the trained classifier while letting direct movie requests guide
the recommendations.

**Implementation:**

- `mood_detector_distilBert.py` now extracts genre requests independently from
  the DistilBERT emotion label. Examples include `funny` / `hilarious` →
  Comedy and `a bit of heart` / `heartfelt` → Drama. It also recognizes common
  requested genres and simple negations such as “I do not want horror.”
- `taste_recommender.py` applies a stronger score boost to genres explicitly
  requested by the user and suppresses the mood-inferred genre boost when an
  explicit genre request is present. Mood-derived avoidance remains in effect,
  except where it conflicts with an explicit positive genre request; explicit
  user avoidance remains strongest.
- `app.py` saves the structured genre intent with the profile's latest intent
  and displays the estimated emotion and explicit requested/avoided genres
  separately, making it clearer what the system extracted.
- Added `tests\test_mood_genre_requests.py` for explicit multi-genre extraction,
  negation, separation from emotion wording, and substring false positives.
- Updated `README.md` and `FILE_OVERVIEW.md`.

**Expected behavior for the reported example:** the model may still report
`surprise` as its emotion estimate; the separate intent extractor should report
`Comedy` and `Drama` as requested genres, and the ranker should prioritize
movies with those genres. The label itself is intentionally not changed to
Comedy.

**Validation:** Tests were added but could not be run in the assistant's
environment because its PowerShell/Python command runner is unavailable. Run
`python -m unittest discover -s tests` in the project environment. Also verify
the live app's detected-genre summary and recommendation cards using the
reported sentence. The phrase extractor is heuristic and may not understand
unlisted or more complex expressions.

**Onboarding follow-up:** The user reported a `KeyError: 'parsed_genres'` while
creating a profile. The onboarding choice generator uses `parsed_genres` only
internally for diversity, then deliberately drops that temporary column before
returning the choices. The onboarding card was incorrectly trying to display
the dropped field. Updated it to parse `genres_list` when rendering each card,
so no internal temporary column is required. Runtime verification is still
needed in the user's app.

### 2026-10-13 — Recorded completed DistilBERT evaluation

The user supplied the final run output, and the generated
`models\distilbert-emotion\evaluation.json` was inspected. The run completed
five epochs; epoch 5 was selected by validation macro-F1 (0.9138). On the
separate 2,000-example test split, accuracy was **0.9230**, macro-F1 **0.8806**,
and weighted F1 **0.9235**. The majority-label baseline scored **0.3475**
accuracy and **0.0860** macro-F1. Full per-class metrics, confusion matrix, and
epoch history are recorded in the fine-tuning entry below. These are public
dataset results, not an evaluation of MoodFlix's end-to-end recommendations.

### 2026-10-13 — DistilBERT public-dataset fine-tuning pipeline

**Need:** Replace the previous use of an already trained emotion model with
actual, measurable fine-tuning while retaining DistilBERT and the six emotion
labels.

**Approach:** Added `train_distilbert_emotion.py`. It loads the `split`
configuration of `dair-ai/emotion`, starts from `distilbert-base-uncased`,
trains a six-label sequence-classification head, selects using validation
macro-F1, and evaluates on the test split once at the end. It uses a fixed seed,
dynamic padding, class-weighted cross-entropy, AdamW, linear learning-rate
decay with warmup, gradient clipping, optional CUDA mixed precision, and
early stopping. Default run settings are:

| Setting | Value |
|---|---:|
| Dataset/config | `dair-ai/emotion` / `split` |
| Train / validation / test examples | 16,000 / 2,000 / 2,000 |
| Labels | 6 |
| Base checkpoint | `distilbert-base-uncased` |
| Default epochs | 5 (early stopping enabled) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Maximum tokens | 128 |
| AdamW weight decay | 0.01 |
| Linear warmup | 10% of scheduled steps |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Checkpoint-selection metric | Validation macro-F1 |

**Expected results:** Fine-tuned weights and tokenizer in
`models\distilbert-emotion`; held-out metrics in
`models\distilbert-emotion\evaluation.json`. Both model artifacts are excluded
from Git by `.gitignore`.

**Integration:** Updated `mood_detector_distilBert.py` to load the local
checkpoint and show a direct setup error if training has not produced a model.
Updated requirements to include Hugging Face Datasets and documented the
training command and limitations in `README.md` and `FILE_OVERVIEW.md`.

**Observed run evidence (completed):**

- The user installed dependencies after encountering
  `ModuleNotFoundError: No module named 'datasets'`.
- The user reported that the dataset loaded and the classification head was
  newly initialized. This is expected when attaching a new six-class head to
  base DistilBERT; the head is subsequently trained.
- During training, the user reported reaching batch 100 of approximately
  1,000 in an epoch and asked about expected runtime and CUDA. That was an
  intermediate progress observation, not a metric.
- The user also saw a PyTorch GradScaler deprecation warning. The script was
  updated to use `torch.amp.GradScaler("cuda", ...)`.
- The user subsequently confirmed all five epochs completed and provided the
  final held-out results. The generated `evaluation.json` was then inspected to
  record configuration, class-wise metrics, the confusion matrix, and baseline.

**Measured metrics:** All metrics below are from the 2,000-example held-out
test split, at the checkpoint selected by validation macro-F1. Values are
rounded to four decimal places from `evaluation.json`.

| Measure | Result |
|---|---:|
| Test accuracy | 0.9230 (92.30%) |
| Test macro-F1 | 0.8806 |
| Test weighted F1 | 0.9235 |
| Test loss | 0.2400 |
| Majority-class baseline accuracy | 0.3475 (34.75%) |
| Majority-class baseline macro-F1 | 0.0860 |
| Best validation accuracy (epoch 5) | 0.9385 (93.85%) |
| Best validation macro-F1 (epoch 5) | 0.9138 |
| Best validation weighted F1 (epoch 5) | 0.9386 |

Per-emotion test results:

| Emotion | Test support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| sadness | 581 | 0.9637 | 0.9587 | 0.9612 |
| joy | 695 | 0.9556 | 0.9295 | 0.9424 |
| love | 159 | 0.7821 | 0.8805 | 0.8284 |
| anger | 275 | 0.9143 | 0.9309 | 0.9225 |
| fear | 224 | 0.8924 | 0.8884 | 0.8904 |
| surprise | 66 | 0.7500 | 0.7273 | 0.7385 |

The confusion matrix is ordered as `[sadness, joy, love, anger, fear,
surprise]` for both rows (actual class) and columns (predicted class):

```text
[[557,  6,  0, 12,  6,  0],
 [  1,646, 39,  4,  0,  5],
 [  0, 18,140,  1,  0,  0],
 [ 11,  3,  0,256,  5,  0],
 [  7,  0,  0,  7,199, 11],
 [  2,  3,  0,  0, 13, 48]]
```

Validation trajectory from `evaluation.json`:

| Epoch | Train loss | Validation accuracy | Validation macro-F1 | Validation weighted F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.7986 | 0.9250 | 0.9006 | 0.9260 |
| 2 | 0.1917 | 0.9295 | 0.9016 | 0.9297 |
| 3 | 0.1320 | 0.9370 | 0.9123 | 0.9373 |
| 4 | 0.0950 | 0.9365 | 0.9105 | 0.9363 |
| 5 | 0.0804 | 0.9385 | 0.9138 | 0.9386 |

**Interpretation:** On this held-out split, the model classified the six
dataset labels substantially better than a classifier that always predicts
the most common label. Class-wise scores are weaker for `surprise` and `love`
than for several larger classes; the surprise test support is only 66. These
numbers are evidence about this dataset and split, not a guarantee that every
MoodFlix request (including slang, mixed emotions, non-English text, or
specific life situations) will be understood correctly. They do not measure
movie recommendation quality.

**Limitations / next action:** Training can be slow on CPU. Check whether
`torch.cuda.is_available()` is true and the GPU is NVIDIA-supported if GPU
training is expected. The saved checkpoint and metrics are local artifacts
under `models\distilbert-emotion`; `.gitignore` excludes them, so preserve or
recreate them when moving to another machine. Next, smoke-test loading this
checkpoint inside MoodFlix and evaluate representative app prompts separately.
The public test score measures the public dataset only, not app recommendation
satisfaction or all real user scenarios.

### 2026-10-12 — Named profiles, voice requests, and feedback

**Need:** Replace the MovieLens sample-user selection with persistent user
preferences, current mood/request input, feedback capture, and eventual
collaborative personalization.

**Approach and implementation:**

- Added `profile_store.py` with a local SQLite database at
  `data\moodflix_profiles.sqlite3`.
- Profile records include a unique case-insensitive name, preferred language,
  preferred genres, optional taste notes, selected favorite movie IDs, and
  structured latest mood intent.
- Feedback records include movie ID, recommendation-appropriateness score
  (1–5), optional movie rating (1–5), pass/watchlist/neutral action, and update
  timestamp. SQLite constraints validate rating and action ranges.
- Reworked `app.py` into a create/select-profile flow. New users choose 10
  titles from 20 varied poster-card choices; returning users select their
  existing profile by name.
- Added text requests and Streamlit audio recording, with editable Groq Whisper
  transcription. Raw request text and audio are not written to the local
  profile database.
- Added `taste_recommender.py` for sparse TF-IDF content scoring, the 2/6/12
  release-year quotas, negative/positive feedback signals, and clustered local
  peer-feedback refinement.
- Kept TMDb for real posters and language metadata. The requested posters are
  not generated by Groq.
- Updated requirements and setup/project documentation.

**Validation:** The assistant's Windows shell runner was unavailable
(`pwsh.exe` and `powershell.exe` were not found), so Python compilation,
SQLite smoke testing, and launching the Streamlit app were not completed by
the assistant at that time. This remains a limitation unless a later test
result is recorded.

**Known product constraints:** The profile database is unencrypted local data.
Collaborative clustering cannot provide a meaningful peer signal until enough
local profiles and feedback exist. MovieLens metadata/ratings are not current
through 2026.

### 2026-10-12 — Project cleanup and documentation

**Need:** Make the codebase easier to navigate and remove old experimental
Python files after identifying what the active application imports.

**Approach:** The active app was rewired away from the old `hybrid.py` and
`mood_recommender.py` integration. The obsolete or superseded scripts removed
were:

- `behavior_tracking.py`
- `feature_engineering.py`
- `final.py`
- `hybrid.py`
- `mood_aware7b.py`
- `mood_detector7a.py`
- `mood_detector_gemini.py`
- `mood_recommender.py`
- `sentiment_classifier.py`

The project retained `mood_detector_distilBert.py`, content-based features,
MovieLens preparation and offline collaborative evaluation. Documentation was
updated to describe active modules and the new profile architecture.

**Validation:** Import-reference searches over active app modules and docs did
not find references to the removed prototypes. Runtime validation was not
available through the assistant's shell.

### Earlier project work — Dataset, recommendation, posters, and setup

The following decisions and work predate the dated implementation entries
above; exact dates and per-change test output were not retained in the current
conversation summary.

1. **Project organization and VS Code run path**
   - Organized setup instructions and project descriptions; added or updated
     requirements and virtual-environment helper scripts.
   - The supported Streamlit entry command is `python -m streamlit run app.py`.
     Passing a non-`.py` extension to Streamlit produced the user's reported
     “requires raw Python (.py) files” error; the fix was to run the actual
     Python entry point.
2. **MovieLens 32M data preparation**
   - Added `download_movielens_32m.py` to download/prepare the GroupLens archive
     and write cleaned rating/movie feature CSVs beneath
     `movielensRatings\ml-32m\cleaned`.
   - Updated `content_based_similarity.py` and the separate
     `collaborative_filtering.py` to use cleaned MovieLens 32M data and sparse
     matrix methods rather than building a dense user-by-item prediction
     matrix.
   - Added an offline temporal ranking-evaluation path reporting
     Precision@K, Recall@K, and NDCG@K. No numerical result from that evaluation
     is recorded here.
3. **Initial preference UI and poster cards**
   - The app previously collected language, genres, free-text preferences,
     MovieLens user ID, and 5–10 selected likes; it showed poster cards and a
     Tinder-inspired Pass/Watchlist deck.
   - This user-ID onboarding was superseded by name-based local profiles and
     exactly ten first-run favorite selections.
4. **TMDb posters, API, and network handling**
   - TMDb was selected for real poster imagery and language metadata associated
     with IMDb IDs. TMDb keys are distinct from Groq keys.
   - The user reported WinError 10054 and certificate verification failures.
     The app was adapted to use `certifi` certificate roots, show optional
     service errors, and continue without posters when requests fail. A TMDb
     outage still means remote images/language data are unavailable.
5. **Release-era mix**
   - The requested proportions became 2 titles before 2000, 6 from 2000–2015,
     and 12 from 2016 onward in a 20-title set. “Recent” is constrained by the
     MovieLens 32M catalog's release-year coverage and is not equivalent to
     releases through 2026.

---

## Current repository map

| File | Current role |
|---|---|
| `app.py` | Streamlit UI, profiles, text/voice requests, poster sources, feedback |
| `profile_store.py` | Local SQLite persistence |
| `taste_recommender.py` | Content ranking and local peer clustering |
| `content_based_similarity.py` | Movie feature loading and sparse TF-IDF model |
| `mood_detector_distilBert.py` | Load fine-tuned local DistilBERT and apply genre/context rules |
| `train_distilbert_emotion.py` | Public dataset fine-tuning and evaluation |
| `collaborative_filtering.py` | Separate MovieLens rating model and ranking metrics |
| `download_movielens_32m.py` | Download and create cleaned MovieLens files |
| `requirements.txt` | Python dependencies |
| `README.md` | Run instructions, configuration, model details |
| `FILE_OVERVIEW.md` | Concise file/data map |
| `tests\` | Focused unit tests for genre intent, persistence, and ranking behavior |
| `models\distilbert-emotion\` | Local generated model artifact; Git-ignored |
| `data\moodflix_profiles.sqlite3` | Local generated user/profile data; Git-ignored |

---

## Current open work

1. **Validate the Streamlit flow and optional services.** Focused unit tests
   and compilation were reported passing by the user after fixes for SQLite
   connection cleanup and the sparse clustering test fixture. Still verify the
   app with TMDb/Groq keys omitted and, if available, with optional services
   enabled.
2. **Test prompts separately from the aggregate dataset metric.** Include
   neutral, explicit cheer-up, breakup, promotion, uncertainty, multilingual
   language requests, and transcription-generated text. Record predicted
   emotion and confidence; do not describe an individual example as proof of
   general accuracy.
3. **Assess domain and language gap.** The public dataset is English social
   text, while the product accepts varied conversational requests. If
   representative, consented MoodFlix examples are later collected, label
   emotion independently of movie satisfaction and preserve a held-out test
   set before considering additional fine-tuning.
4. **Validate app flow and UX.** Verify returning profile selection, exactly
   ten onboarding likes, TMDb-to-Wikipedia poster fallback, feedback
   persistence, watchlist rendering, and behavior with too few local profiles
   for clustering.
