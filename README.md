# MoodFlix

MoodFlix is a local, mood-aware movie recommender built with Streamlit. It uses a
MovieLens 32M catalog, a content-based TF-IDF model, DistilBERT emotion analysis,
optional Groq Whisper transcription, TMDb/Wikipedia poster lookup, and feedback
from local named profiles.


The first training run downloads the public `dair-ai/emotion` **split**
configuration (16,000 training, 2,000 validation, and 2,000 test examples) and
base `distilbert-base-uncased` weights. It fine-tunes a six-class emotion head,
selects the checkpoint with the highest validation macro-F1, and then evaluates
the held-out test split once. The fine-tuned model and `evaluation.json` are
saved under `models\distilbert-emotion` (excluded from Git). Training defaults
to five epochs with early stopping; it uses CUDA when available and CPU
otherwise. CPU training may take substantial time. Re-run the training command
to retrain; the app loads the saved local fine-tuned model and does not silently
fall back to a different pretrained emotion checkpoint.

The completed five-epoch run recorded **92.30% held-out test accuracy** and
**0.8806 macro-F1** (weighted F1: **0.9235**) on this dataset's 2,000-example
test split. The majority-class baseline scored 34.75% accuracy and 0.0860
macro-F1. These are dataset metrics, not a guarantee for every MoodFlix prompt
or a measure of recommendation quality; see `evaluation.json` for per-class
metrics and the confusion matrix.


## Run with Docker

Install and start Docker Desktop, then build the image and start MoodFlix from
the project folder

## First use and returning profiles

Create a profile with a name (profiles are local and do not use passwords). Set
your usual language and genres, add optional taste notes, and select exactly 10
favorites from a randomly varied set of 20 poster cards. Starter choices and
recommendations target 2 titles before 2000, 6 from 2000–2015, and 12 from
2016 onward. The MovieLens 32M catalog covers ratings through October 2023, so
“recent” means 2016 onward within that catalog, not releases through 2026.

On later visits, choose the saved profile by name. Type a current movie request
or record it with the microphone. The app sends recorded audio to Groq for
Whisper transcription; it does not save the audio or raw request text. The
transcription can be edited before recommendations are generated. For each
movie, score how well it matched the request, optionally rate the film if you
have seen it, and choose whether to pass or add it to your watchlist.

Profile settings, favorite movie IDs, structured mood labels, and recommendation
feedback are stored locally in `data\moodflix_profiles.sqlite3`. This is
personal, unencrypted local data; anyone with access to this computer and file
can read it. The app does not store audio or raw mood text. Collaborative
recommendations use K-means clusters of local profile movie tastes and feedback;
until similar profiles and feedback exist, recommendations remain content-based.

## API keys

The sidebar accepts keys at runtime, or you can set them in the VS Code
PowerShell terminal before starting Streamlit

Create a TMDb v3 key in your TMDb account's API settings. It is used to find
posters and original-language metadata. If TMDb is not configured, unavailable,
or has no poster for a title, MoodFlix searches English Wikipedia for a page
thumbnail without requiring an API key; results may be missing or may show an
image other than the official poster. Groq's key is used only for
`whisper-large-v3-turbo` speech transcription. Keys typed in the app are not
written into the project. Do not put API keys in source files or commit them.
If a key was previously pasted into a chat or shared, revoke it and use a new
key.

The app continues without keys: Wikipedia images are attempted for movie
cards, but language filtering requires TMDb and voice transcription requires
Groq. If no image is available from either source, the card uses its text-only
fallback. Wikipedia's thumbnail coverage and exact match quality are not
guaranteed.

## Recommendation model and limitations

- **Content first:** TF-IDF movie feature vectors rank titles against the
  profile's 10 favorites, saved feedback, current request, and genre choices.
- **Collaborative refinement:** with at least three usable local profiles,
  K-means groups profile taste vectors and feedback from same-cluster profiles
  gives a secondary ranking signal. Sparse/new profiles fall back to content
  ranking.
- **Mood analysis:** DistilBERT is fine-tuned from the general-purpose
  `distilbert-base-uncased` checkpoint on the public English `dair-ai/emotion`
  dataset (sadness, joy, love, anger, fear, surprise). Training records test
  accuracy, macro/weighted F1, per-class scores, a confusion matrix, and a
  majority-class baseline in `models\distilbert-emotion\evaluation.json`.
  This measures performance on that held-out public dataset, not every
  situation or language. It is not fine-tuned on MoodFlix users: the project
  has no representative, human-labeled mood dataset, and movie ratings are
  not valid emotion labels. Curated context rules supplement the model.
- **Explicit genre intent:** the six emotion labels do not include movie
  genres. MoodFlix separately extracts common explicit genre requests (for
  example, “funny” → Comedy and “a bit of heart” → Drama) and ranks those
  requested genres ahead of inferred mood genres. Negated requests such as
  “I do not want horror” are treated as genres to avoid. This lightweight
  phrase matcher is not a trained language model and may miss unfamiliar
  wording.
- **Offline evaluation:** after preparing the ratings CSV, compare the separate
  MovieLens collaborative model with:

  It reports Precision@10, Recall@10, and NDCG@10 on a sampled temporal split;
  this evaluates the MovieLens model, not real-world satisfaction of the
  profile-based Streamlit app.

This product uses the TMDb API but is not endorsed or certified by TMDb.

For the implementation history, decisions, training configuration, validation
status, and next steps, see [PROJECT_LOG.md](./PROJECT_LOG.md).
