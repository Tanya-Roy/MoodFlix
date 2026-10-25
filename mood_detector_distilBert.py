"""Local six-emotion DistilBERT classifier with curated movie-genre rules.

The fine-tuned model estimates sadness, joy, love, anger, fear, and surprise.
The rules translate those labels and a small set of common phrases into
preferred/avoided genres. Its English-language predictions and hand-written
rules do not cover every mood or situation.
"""

from pathlib import Path
import re

from transformers import pipeline

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "distilbert-emotion"
VALID_GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                 "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                 "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
GENRE_REQUEST_PHRASES = {
    "Action": ("action", "action-packed"),
    "Adventure": ("adventure", "adventurous"),
    "Animation": ("animation", "animated", "cartoon"),
    "Children's": ("kids movie", "children's movie", "childrens movie"),
    "Comedy": ("comedy", "funny", "hilarious", "make me laugh", "something lighthearted"),
    "Crime": ("crime", "detective story"),
    "Documentary": ("documentary", "docuseries"),
    "Drama": ("drama", "heartfelt", "with heart", "a bit of heart"),
    "Fantasy": ("fantasy", "magical"),
    "Film-Noir": ("film noir", "film-noir"),
    "Horror": ("horror", "scary", "frightening"),
    "Musical": ("musical", "with songs"),
    "Mystery": ("mystery", "whodunit"),
    "Romance": ("romance", "romantic", "love story"),
    "Sci-Fi": ("sci-fi", "sci fi", "science fiction"),
    "Thriller": ("thriller", "suspenseful", "suspense"),
    "War": ("war movie", "war film"),
    "Western": ("western", "cowboy movie"),
}
NEGATION_CUES = (
    "not",
    "no",
    "never",
    "without",
    "avoid",
    "excluding",
    "except",
    "don't want",
    "do not want",
    "not looking for",
    "not in the mood for",
)

# What each emotion WANTS to watch, split into two cases:
#   "match_mood": genres that lean into the feeling
#   "cheer_up":   genres to offer instead, if the person signals they
#                 want OUT of the feeling rather than into it
EMOTION_TO_GENRES = {
    "sadness":  {"match_mood": ["Drama", "Romance"],              "cheer_up": ["Comedy", "Animation"]},
    "joy":      {"match_mood": ["Comedy", "Adventure", "Musical"], "cheer_up": ["Comedy", "Adventure"]},
    "love":     {"match_mood": ["Romance", "Drama"],               "cheer_up": ["Romance", "Comedy"]},
    "anger":    {"match_mood": ["Action", "Crime", "Thriller"],    "cheer_up": ["Comedy", "Adventure"]},
    "fear":     {"match_mood": ["Horror", "Thriller", "Mystery"],  "cheer_up": ["Comedy", "Animation"]},
    "surprise": {"match_mood": ["Mystery", "Sci-Fi", "Fantasy"],   "cheer_up": ["Adventure", "Comedy"]},
}

# Genres that clash with each emotion -- used for avoid_genres regardless
# of match_mood/cheer_up, e.g. don't suggest Horror to someone who is sad.
EMOTION_TO_AVOID = {
    "sadness":  ["Horror", "War"],
    "joy":      ["Horror", "War"],
    "love":     ["Horror", "War"],
    "anger":    ["Horror"],
    "fear":     ["Horror", "Thriller", "War"],
    "surprise": ["Documentary"],
}

# Phrases that mean "I genuinely don't know what I want" -- checked
# BEFORE trusting the emotion classifier's confident label.
UNCERTAINTY_PHRASES = [
    "i don't know", "i dont know", "idk", "not sure", "no idea",
    "whatever", "anything is fine", "don't know what i want",
    "dont know what i want", "don't care", "dont care",
]

# Phrases that mean "I feel bad, but I want to feel BETTER, not wallow" --
# this flips match_mood genres into cheer_up genres.
CHEER_UP_PHRASES = [
    "feel better", "cheer me up", "cheer up", "take my mind off",
    "make me feel better", "something to distract", "get my mind off",
    "move on", "forget about it",
]

# Phrases that NEGATE a mood ("not in the mood for anything sad"). The
# emotion classifier has no concept of negation -- it will still see
# "sad" and say sadness. Treating a negated mood as an implicit
# cheer_up signal means we actively avoid the genres matching the
# REJECTED mood, instead of recommending them.
NEGATION_PHRASES = [
    "not in the mood for", "don't want", "dont want", "not feeling",
    "no interest in", "not looking for", "not interested in",
]

# A small set of hand-picked context rules, not a comprehensive rule set.
# Each rule: (phrases to match, genres to avoid, genres to add as preferred)
CONTEXT_RULES = [
    (["broke up", "break up", "breakup", "ex-boyfriend", "ex-girlfriend",
      "ex boyfriend", "ex girlfriend", "heartbroken", "heartbreak"],
     ["Romance"], []),

    (["passed away", "someone died", "lost my dad", "lost my mom",
      "lost my mother", "lost my father", "funeral", "my dog died",
      "my cat died", "my pet died"],
     ["War"], []),

    (["exam tomorrow", "interview tomorrow", "test tomorrow",
      "nervous about", "anxious about", "scared about tomorrow"],
     ["Thriller", "Horror", "War"], []),

    (["just had a baby", "newborn", "up all night with the baby", "new parent"],
     ["Horror", "Thriller"], ["Children's", "Animation"]),

    (["got robbed", "someone broke into my house", "car accident",
      "break-in", "break in"],
     ["Crime", "Thriller", "Horror"], []),
]

INTENSITY_BY_EMOTION = {
    "sadness": "low", "joy": "high", "love": "medium",
    "anger": "high", "fear": "high", "surprise": "medium",
}

_classifier = None


def _get_classifier():
    """Load the locally fine-tuned model on first use."""
    global _classifier
    if _classifier is None:
        if not (MODEL_DIR / "config.json").is_file():
            raise FileNotFoundError(
                f"Fine-tuned DistilBERT checkpoint not found at {MODEL_DIR}. "
                "Run `python train_distilbert_emotion.py` first."
            )
        print(f"Loading fine-tuned local emotion model from {MODEL_DIR}...")
        _classifier = pipeline(
            "text-classification",
            model=str(MODEL_DIR),
            tokenizer=str(MODEL_DIR),
            top_k=None,
        )
    return _classifier


def _contains_any(text, phrases):
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


def _has_word_or_phrase(text, phrase):
    return re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        text.casefold(),
    ) is not None


def _is_negated_genre_request(text: str, phrase: str) -> bool:
    """Recognize common short negations immediately before a genre phrase."""
    match = re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        text.casefold(),
    )
    if match is None:
        return False
    prefix = text.casefold()[max(0, match.start() - 36):match.start()]
    prefix = re.split(
        r"[.!?;]|,\s*|\bbut\b|\bhowever\b|\binstead\b",
        prefix,
    )[-1]
    return any(
        re.search(rf"(?:^|\W){re.escape(cue)}(?:\W|$)", prefix)
        for cue in NEGATION_CUES
    )


def extract_genre_requests(user_text: str) -> tuple[list[str], list[str]]:
    """Extract explicit preferred/avoided movie genres separately from emotion."""
    preferred = []
    avoided = []
    for genre, phrases in GENRE_REQUEST_PHRASES.items():
        mentions = [
            phrase
            for phrase in phrases
            if _has_word_or_phrase(user_text, phrase)
        ]
        if not mentions:
            continue
        if any(_is_negated_genre_request(user_text, phrase) for phrase in mentions):
            avoided.append(genre)
        else:
            preferred.append(genre)
    return preferred, avoided


def _apply_context_rules(user_text):
    """Returns (avoid_add, preferred_add) -- genres the five specific
    real-life situations in CONTEXT_RULES imply, on top of whatever the
    emotion classifier and cheer_up/uncertainty/negation heuristics
    already decided. These rules are deliberately limited."""
    avoid_add, preferred_add = [], []
    for phrases, avoid_genres_rule, preferred_genres_rule in CONTEXT_RULES:
        if _contains_any(user_text, phrases):
            avoid_add.extend(avoid_genres_rule)
            preferred_add.extend(preferred_genres_rule)
    return avoid_add, preferred_add


def get_mood_profile(user_text, confidence_threshold=0.4):
    """Infer emotion and extract explicit movie-genre requests from user text.

    confidence_threshold is the minimum emotion score required to return
    mood-derived genres. Explicit genre requests are reported separately.
    """
    clf = _get_classifier()
    requested_genres, avoided_requested_genres = extract_genre_requests(user_text)

    # --- Step 1: keyword pre-checks, BEFORE we even look at the model ---
    is_uncertain = _contains_any(user_text, UNCERTAINTY_PHRASES)
    is_negated_mood = _contains_any(user_text, NEGATION_PHRASES)

    # A negated mood ("not in the mood for anything sad") is treated as
    # an implicit cheer_up signal -- same mechanism as an explicit
    # "cheer me up" phrase.
    wants_cheer_up = _contains_any(user_text, CHEER_UP_PHRASES) or is_negated_mood

    # Apply the small set of hand-written context rules.
    context_avoid_add, context_preferred_add = _apply_context_rules(user_text)

    # --- Step 2: run the actual emotion classifier ---
    raw = clf(user_text)[0]  # top_k=None -> list of {"label":..., "score":...} for all 6
    raw_sorted = sorted(raw, key=lambda d: d["score"], reverse=True)
    top = raw_sorted[0]
    mood_label = top["label"]
    top_score = top["score"]

    scores = {d["label"]: round(d["score"], 3) for d in raw_sorted}

    emotion_map = EMOTION_TO_GENRES.get(mood_label, {"match_mood": [], "cheer_up": []})

    avoid_genres = EMOTION_TO_AVOID.get(mood_label, [])
    avoid_genres = list(dict.fromkeys(avoid_genres + context_avoid_add))
    if is_negated_mood:
        # The classifier still says e.g. "sadness" for a NEGATED sad mood --
        # it has no concept of negation. So explicitly avoid the genres
        # that would have matched the rejected mood, rather than
        # recommending them.
        avoid_genres = list(dict.fromkeys(avoid_genres + emotion_map["match_mood"]))

    # --- Step 3: decide what to actually return ---
    if is_uncertain:
        return {
            "mood_label": mood_label,
            "wants": "unclear",
            "preferred_genres": [],
            "requested_genres": requested_genres,
            "mood_avoid_genres": avoid_genres,
            "requested_avoid_genres": avoided_requested_genres,
            "avoid_genres": list(
                dict.fromkeys(avoid_genres + avoided_requested_genres)
            ),
            "intensity": INTENSITY_BY_EMOTION.get(mood_label, "medium"),
            "low_confidence": True,
            "reason": "uncertainty phrase detected in input; emotion classifier's genre "
                       "guess was suppressed on purpose, not because the model failed",
            "_emotion_scores": scores,
        }

    if top_score < confidence_threshold:
        return {
            "mood_label": mood_label,
            "wants": "unclear",
            "preferred_genres": [],
            "requested_genres": requested_genres,
            "mood_avoid_genres": avoid_genres,
            "requested_avoid_genres": avoided_requested_genres,
            "avoid_genres": list(
                dict.fromkeys(avoid_genres + avoided_requested_genres)
            ),
            "intensity": INTENSITY_BY_EMOTION.get(mood_label, "medium"),
            "low_confidence": True,
            "reason": f"top emotion '{mood_label}' only scored {top_score:.2f}, "
                       f"below the {confidence_threshold} confidence threshold",
            "_emotion_scores": scores,
        }

    wants = "cheer_up" if wants_cheer_up else "match_mood"

    if context_preferred_add:
        # A context rule (e.g. new parent) is a more specific, reliable
        # signal than the raw emotion label -- it OVERRIDES the
        # emotion-based guess entirely instead of blending with it
        # (blending produced a contradictory Drama/Romance +
        # Children's/Animation mix on a real test input).
        preferred_genres = list(dict.fromkeys(context_preferred_add))
    else:
        preferred_genres = emotion_map[wants]

    # Consistency invariant: avoid_genres always wins. Without this, two
    # independently-correct pieces of logic (emotion-based preferred_genres
    # and a context/negation-based avoid_genres) could still both claim
    # the same genre at once -- which happened for real on the
    # "exam tomorrow, anxious" test case (Horror/Thriller in both lists).
    preferred_genres = [g for g in preferred_genres if g not in avoid_genres]
    requested_genres = [
        genre for genre in requested_genres if genre not in avoided_requested_genres
    ]

    return {
        "mood_label": mood_label,
        "wants": wants,
        "preferred_genres": preferred_genres,
        "requested_genres": requested_genres,
        "mood_avoid_genres": avoid_genres,
        "requested_avoid_genres": avoided_requested_genres,
        "avoid_genres": list(
            dict.fromkeys(avoid_genres + avoided_requested_genres)
        ),
        "intensity": INTENSITY_BY_EMOTION.get(mood_label, "medium"),
        "low_confidence": False,
        "reason": f"emotion classifier detected '{mood_label}' ({top_score:.2f} confidence)"
                   + (", negated mood detected -- treated as implicit cheer_up"
                      if is_negated_mood else "")
                   + (", cheer-up phrase detected so genres were flipped away from the mood"
                      if wants_cheer_up and not is_negated_mood else "")
                   + (", context rule applied (see CONTEXT_RULES)"
                      if context_avoid_add or context_preferred_add else ""),
        "_emotion_scores": scores,
    }


if __name__ == "__main__":
    test_inputs = [
        "today was a shit day i want something lowkey and chill",
        "i just got promoted!! give me something fun and exciting",
        "i need a good cry, something emotional and sad",
        "not in the mood for anything sad",
        "asdf hello",
        "something funny but with a bit of heart",
        # the hard case that exposed the mood-vs-wants gap:
        "i just feel weird. we broke up yesterday. i wanna watch something which makes me feel better",
        # literal uncertainty, no emotional content to latch onto at all:
        "i don't know what i want to watch",
        # Context rule cases:
        "my dad passed away last week, i don't know what to watch",
        "i have a big exam tomorrow and i'm so anxious",
        "we just had a baby and i'm exhausted, want something to watch tonight",
    ]

    for text in test_inputs:
        print(f"Input: '{text}'")
        profile = get_mood_profile(text)
        print(f"  mood_label: {profile['mood_label']}   wants: {profile['wants']}")
        print(f"  preferred_genres: {profile['preferred_genres']}")
        print(f"  requested_genres: {profile['requested_genres']}")
        print(f"  requested_avoid_genres: {profile['requested_avoid_genres']}")
        print(f"  avoid_genres: {profile['avoid_genres']}")
        print(f"  intensity: {profile['intensity']}")
        print(f"  low_confidence: {profile['low_confidence']}")
        print(f"  reason: {profile['reason']}")
        print(f"  (raw emotion scores: {profile['_emotion_scores']})")
        print()