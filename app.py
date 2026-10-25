"""MoodFlix: named local profiles and content-first movie recommendations."""

import json
import os
import random
import ssl
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd
import streamlit as st
from groq import APIError, Groq

from content_based_similarity import ContentBasedRecommender
from mood_detector_distilBert import get_mood_profile
from profile_store import (
    initialize_store,
    list_profiles,
    load_feedback,
    load_profile,
    save_feedback,
    save_latest_intent,
    save_profile,
)
from taste_recommender import (
    ERA_BUCKETS,
    balanced_recommendations,
    parse_genres,
    rank_recommendations,
)


TMDB_API_URL = "https://api.themoviedb.org/3/find"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
HTTPS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
LANGUAGES = {
    "Any language": None,
    "English": "en",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn",
    "Marathi": "mr",
    "Spanish": "es",
    "French": "fr",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
}
LANGUAGE_ALIASES = {
    "English": ("english",),
    "Hindi": ("hindi",),
    "Malayalam": ("malayalam", "malayali"),
    "Tamil": ("tamil",),
    "Telugu": ("telugu",),
    "Kannada": ("kannada",),
    "Bengali": ("bengali",),
    "Marathi": ("marathi",),
    "Spanish": ("spanish",),
    "French": ("french",),
    "Japanese": ("japanese",),
    "Korean": ("korean",),
    "Chinese": ("chinese", "mandarin"),
}

st.set_page_config(page_title="MoodFlix", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #342238 0%, #15151d 48%, #101116 100%);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(255, 255, 255, 0.13);
        border-radius: 22px;
        background: rgba(30, 29, 39, 0.84);
    }
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(100deg, #ff4d6d, #ff784f);
        border: 0;
        color: white;
        font-weight: 700;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 999px;
        min-height: 2.8rem;
    }
    button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 3px solid #ffd166 !important;
        outline-offset: 2px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the movie catalog and content model...")
def load_content_model() -> ContentBasedRecommender:
    return ContentBasedRecommender()


@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_info(imdb_id: str, api_key: str) -> dict:
    query = urlencode({"external_source": "imdb_id", "api_key": api_key})
    request = Request(
        f"{TMDB_API_URL}/{imdb_id}?{query}",
        headers={"Accept": "application/json"},
    )
    with urlopen(request, timeout=8, context=HTTPS_CONTEXT) as response:
        result = json.loads(response.read().decode("utf-8"))
    movies = result.get("movie_results", [])
    if not movies:
        return {}
    return {
        "poster_path": movies[0].get("poster_path"),
        "original_language": movies[0].get("original_language"),
    }


@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_poster(poster_path: str) -> bytes:
    request = Request(
        f"{TMDB_IMAGE_URL}{poster_path}",
        headers={"User-Agent": "MoodFlix movie recommendation app"},
    )
    with urlopen(request, timeout=10, context=HTTPS_CONTEXT) as response:
        return response.read()


@st.cache_data(ttl=86400, show_spinner=False)
def get_wikipedia_poster(title: str, release_year: int | None = None) -> bytes | None:
    search_term = f"{title} film"
    if release_year:
        search_term += f" {release_year}"
    query = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": search_term,
            "gsrnamespace": 0,
            "gsrlimit": 5,
            "prop": "pageimages",
            "piprop": "thumbnail",
            "pithumbsize": 500,
            "format": "json",
        }
    )
    request = Request(
        f"{WIKIPEDIA_API_URL}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "MoodFlix/1.0 (local movie recommendation app)",
        },
    )
    with urlopen(request, timeout=8, context=HTTPS_CONTEXT) as response:
        result = json.loads(response.read().decode("utf-8"))

    pages = result.get("query", {}).get("pages", {}).values()
    ranked_pages = sorted(pages, key=lambda page: page.get("index", 0))
    thumbnail_url = next(
        (
            page.get("thumbnail", {}).get("source")
            for page in ranked_pages
            if page.get("thumbnail", {}).get("source")
        ),
        None,
    )
    if not thumbnail_url:
        return None

    image_request = Request(
        thumbnail_url,
        headers={"User-Agent": "MoodFlix/1.0 (local movie recommendation app)"},
    )
    with urlopen(image_request, timeout=10, context=HTTPS_CONTEXT) as response:
        return response.read()


def report_tmdb_error(error: Exception) -> None:
    st.session_state["tmdb_unavailable"] = True
    if st.session_state.get("tmdb_error_shown"):
        return
    if isinstance(error, HTTPError):
        message = f"TMDb returned HTTP {error.code}: {error.reason}."
        if error.code in (401, 403):
            message += " Check the TMDb API v3 key."
    elif isinstance(error, URLError):
        message = f"TMDb could not be reached: {error.reason}"
    else:
        message = f"TMDb request failed: {error}"
    st.warning(
        message
        + " Wikipedia will still be tried for poster images; recommendations "
        "continue without TMDb language metadata."
    )
    st.session_state["tmdb_error_shown"] = True


def report_wikipedia_error(error: Exception) -> None:
    st.session_state["wikipedia_unavailable"] = True
    if st.session_state.get("wikipedia_error_shown"):
        return
    st.warning(
        f"Wikipedia poster lookup failed: {error}. "
        "Recommendations will continue with title-only cards."
    )
    st.session_state["wikipedia_error_shown"] = True


def poster_for_movie(movie: pd.Series | dict, api_key: str) -> bytes | None:
    imdb_id = movie.get("tconst")
    if (
        api_key
        and not st.session_state.get("tmdb_unavailable")
        and isinstance(imdb_id, str)
        and imdb_id.startswith("tt")
    ):
        try:
            poster_path = get_tmdb_info(imdb_id, api_key).get("poster_path")
            if poster_path:
                return get_tmdb_poster(poster_path)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            report_tmdb_error(error)

    if st.session_state.get("wikipedia_unavailable"):
        return None
    title = movie.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    year = pd.to_numeric(movie.get("release_year"), errors="coerce")
    release_year = int(year) if pd.notna(year) else None
    try:
        return get_wikipedia_poster(title.strip(), release_year)
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        report_wikipedia_error(error)
        return None


def diverse_movie_choices(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Select 20 randomized titles with low genre overlap and the 2/6/12 era mix."""
    randomizer = random.Random(seed)
    selected = []
    for _, start_year, end_year, quota in ERA_BUCKETS:
        years = pd.to_numeric(items["release_year"], errors="coerce")
        if end_year is None:
            in_era = years >= start_year
        else:
            in_era = years.between(start_year, end_year, inclusive="both")
        pool = items.loc[in_era & items["tconst"].notna()].copy()
        pool = pool.drop_duplicates("item_id")
        if len(pool) < quota:
            raise ValueError(
                f"The catalog has only {len(pool)} movies for the {start_year} era; "
                "20 varied onboarding choices are not available."
            )
        pool = pool.sample(n=min(350, len(pool)), random_state=seed)
        pool["parsed_genres"] = pool["genres_list"].map(parse_genres)
        remaining = pool.to_dict("records")
        chosen = []
        while remaining and len(chosen) < quota:
            def diversity(candidate):
                candidate_genres = set(candidate["parsed_genres"])
                similarities = []
                for existing in chosen:
                    existing_genres = set(existing["parsed_genres"])
                    union = candidate_genres | existing_genres
                    similarities.append(
                        len(candidate_genres & existing_genres) / len(union)
                        if union
                        else 0.0
                    )
                overlap = max(similarities, default=0.0)
                return (1.0 - overlap) + randomizer.random() * 0.2

            best = max(range(len(remaining)), key=lambda index: diversity(remaining[index]))
            chosen.append(remaining.pop(best))
        selected.extend(chosen)

    randomizer.shuffle(selected)
    return pd.DataFrame(selected).drop(columns=["parsed_genres"], errors="ignore")


def create_profile_screen(content_model, tmdb_api_key: str) -> None:
    onboarding_name = st.session_state.get("onboarding_name")
    if onboarding_name:
        onboarding_screen(content_model, onboarding_name, tmdb_api_key)
        if st.button("Cancel profile setup"):
            st.session_state.pop("onboarding_name", None)
            st.session_state.pop("onboarding_seed", None)
            st.rerun()
        return

    st.subheader("Welcome to MoodFlix")
    st.write("Create a local profile or continue with one saved on this computer.")
    names = list_profiles()
    options = ["Create a new profile"] + names
    choice = st.selectbox("Choose a profile", options, key="profile_choice")

    if choice == "Create a new profile":
        name = st.text_input("Your name or profile name", max_chars=60)
        if st.button("Create profile", type="primary"):
            if not name.strip():
                st.error("Enter a name for this profile.")
            elif load_profile(name):
                st.error("That name already has a profile. Select it from the list.")
            else:
                st.session_state["onboarding_name"] = name.strip()
                st.session_state["onboarding_seed"] = random.randrange(1, 2**31)
                st.rerun()
    else:
        if st.button(f"Continue as {choice}", type="primary"):
            profile = load_profile(choice)
            if profile is None:
                st.error("That profile could not be loaded. Refresh and try again.")
            else:
                st.session_state["active_profile_id"] = profile["profile_id"]
                st.session_state["active_profile_name"] = profile["name"]
                st.rerun()

def onboarding_screen(content_model, name: str, tmdb_api_key: str) -> None:
    st.divider()
    st.subheader(f"Let's set up {name}'s taste")
    st.write(
        "Choose your usual preferences, then pick **exactly 10** movies you already like. "
        "The 20 movie choices are varied and span different release eras."
    )
    st.caption(
        "Both the starter choices and recommendations aim for 2 movies before 2000, "
        "6 from 2000–2015, and 12 from 2016 onward."
    )

    seed = st.session_state["onboarding_seed"]
    choices_key = f"onboarding_choices_{seed}"
    if choices_key not in st.session_state:
        try:
            st.session_state[choices_key] = diverse_movie_choices(
                content_model.items, seed
            )
        except ValueError as error:
            st.error(str(error))
            return
    choices = st.session_state[choices_key]

    genres = sorted(
        {
            genre
            for values in content_model.items["genres_list"].map(parse_genres)
            for genre in values
        }
    )
    with st.form("profile_preferences"):
        language = st.selectbox("Preferred language", list(LANGUAGES))
        preferred_genres = st.multiselect("Genres you usually enjoy", genres)
        other_preferences = st.text_area(
            "Other preferences",
            placeholder="e.g. clever mysteries, family-friendly stories, or slow-burn dramas",
            height=80,
        )
        st.markdown("#### Choose your 10 favorite movies")
        columns = st.columns(4)
        for position, (_, movie) in enumerate(choices.iterrows()):
            with columns[position % 4]:
                with st.container(border=True):
                    image = poster_for_movie(movie, tmdb_api_key)
                    if image:
                        st.image(
                            image,
                            caption=movie["title"],
                            use_container_width=True,
                        )
                    else:
                        st.markdown("## 🎬")
                        st.caption("Poster unavailable")
                    year = movie.get("release_year")
                    year_label = str(int(year)) if pd.notna(year) else "Year unknown"
                    st.markdown(f"**{movie['title']}**")
                    movie_genres = parse_genres(movie.get("genres_list", []))
                    st.caption(
                        f"{year_label} · "
                        f"{', '.join(movie_genres) or 'Genre unknown'}"
                    )
                    st.checkbox(
                        f"I like {movie['title']}",
                        key=f"onboard_like_{seed}_{int(movie['item_id'])}",
                    )
        submitted = st.form_submit_button(
            "Save profile and find movies",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        liked_ids = [
            int(movie["item_id"])
            for _, movie in choices.iterrows()
            if st.session_state.get(
                f"onboard_like_{seed}_{int(movie['item_id'])}", False
            )
        ]
        if len(liked_ids) != 10:
            st.error(f"Select exactly 10 movies; you selected {len(liked_ids)}.")
            return
        profile_id = save_profile(
            name,
            language,
            preferred_genres,
            other_preferences,
            liked_ids,
        )
        st.session_state["active_profile_id"] = profile_id
        st.session_state["active_profile_name"] = name
        st.session_state.pop("onboarding_name", None)
        st.session_state.pop("recommendations", None)
        st.rerun()


def transcribe_recording(audio_file, groq_api_key: str) -> str:
    if not groq_api_key:
        raise ValueError("Enter a Groq API key in the sidebar to transcribe audio.")
    if audio_file is None:
        raise ValueError("Record audio before requesting transcription.")
    client = Groq(api_key=groq_api_key)
    transcription = client.audio.transcriptions.create(
        file=(audio_file.name or "mood-recording.wav", audio_file.getvalue()),
        model="whisper-large-v3-turbo",
        response_format="json",
    )
    return transcription.text.strip()


def requested_language_from_text(text: str) -> str | None:
    lowered = text.casefold()
    for language, aliases in LANGUAGE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return language
    script_languages = (
        ("\u0d00", "\u0d7f", "Malayalam"),
        ("\u0b80", "\u0bff", "Tamil"),
        ("\u0c00", "\u0c7f", "Telugu"),
        ("\u0c80", "\u0cff", "Kannada"),
        ("\u0980", "\u09ff", "Bengali"),
        ("\u0900", "\u097f", "Hindi"),
    )
    for first, last, language in script_languages:
        if any(first <= character <= last for character in text):
            return language
    return None


def show_active_profile(content_model, tmdb_api_key: str, groq_api_key: str) -> None:
    profile_name = st.session_state["active_profile_name"]
    profile = load_profile(profile_name)
    if profile is None:
        st.error("This profile is missing from the local database. Switch profiles.")
        return
    profile["feedback"] = load_feedback(profile["profile_id"])

    title_col, action_col = st.columns([4, 1])
    with title_col:
        st.subheader(f"Good to see you, {profile_name}!")
        st.caption(
            f"Usual language: {profile['preferred_language']} · "
            f"Favorite genres: {', '.join(profile['preferred_genres']) or 'Not specified'}"
        )
    with action_col:
        if st.button("Switch profile"):
            for key in (
                "active_profile_id",
                "active_profile_name",
                "recommendations",
                "recommendation_index",
                "movie_request",
                "transcribed_request",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    st.markdown("### What are you in the mood for?")
    st.caption(
        "Type or record your request. Audio is sent to Groq for transcription and "
        "is not saved; only the derived mood and preference labels are retained."
    )
    audio = st.audio_input("Record your movie request")
    if st.button("Transcribe with Groq Whisper", disabled=audio is None):
        try:
            with st.spinner("Transcribing your recording..."):
                st.session_state["transcribed_request"] = transcribe_recording(
                    audio, groq_api_key
                )
                st.session_state["movie_request"] = st.session_state[
                    "transcribed_request"
                ]
        except (APIError, OSError, ValueError) as error:
            st.error(f"Voice transcription failed: {error}")

    request_text = st.text_area(
        "Your request (you can edit the transcription)",
        key="movie_request",
        placeholder="e.g. I'm heartbroken and want a Malayalam movie that will cheer me up",
        height=100,
    )
    if st.button("Find my matches", type="primary", use_container_width=True):
        if not request_text.strip():
            st.error("Tell MoodFlix what kind of movie you want today.")
            return
        try:
            with st.spinner("Understanding your request and ranking movies..."):
                try:
                    mood = get_mood_profile(request_text.strip())
                except (ImportError, OSError) as error:
                    st.error(f"Could not load the DistilBERT mood model: {error}")
                    st.info(
                        "Train the local model first with "
                        "`python train_distilbert_emotion.py` from the project folder."
                    )
                    return
                requested_language = requested_language_from_text(request_text)
                mood["requested_language"] = requested_language
                language = requested_language or profile["preferred_language"]
                language_lookup = None
                if LANGUAGES.get(language) and tmdb_api_key:
                    language_lookup = lambda imdb_id: get_tmdb_info(
                        imdb_id, tmdb_api_key
                    )
                recommendation_query = " ".join(
                    value
                    for value in (
                        profile["other_preferences"].strip(),
                        request_text.strip(),
                    )
                    if value
                )
                try:
                    recommendations, peer_count, language_filtered = rank_recommendations(
                        content_model,
                        profile,
                        mood,
                        recommendation_query,
                        language,
                        LANGUAGES,
                        tmdb_info_lookup=language_lookup,
                    )
                except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                    report_tmdb_error(error)
                    recommendations, peer_count, language_filtered = rank_recommendations(
                        content_model,
                        profile,
                        mood,
                        recommendation_query,
                        language,
                        LANGUAGES,
                    )
                if not language_filtered and LANGUAGES.get(language):
                    if not tmdb_api_key:
                        st.info(
                            "Language filtering needs a TMDb API v3 key. "
                            "Personalized results are shown without language filtering."
                        )
                    elif not st.session_state.get("tmdb_unavailable"):
                        st.info(
                            "There were not enough titles in that language to maintain "
                            "the full release-era mix, so the results use your other "
                            "preferences instead."
                        )
                save_latest_intent(
                    profile["profile_id"],
                    {
                        "mood": mood.get("mood_label"),
                        "wants": mood.get("wants"),
                        "preferred_genres": mood.get("preferred_genres", []),
                        "requested_genres": mood.get("requested_genres", []),
                        "requested_avoid_genres": mood.get(
                            "requested_avoid_genres", []
                        ),
                        "avoid_genres": mood.get("avoid_genres", []),
                        "requested_language": requested_language,
                    },
                )
                st.session_state["last_request_summary"] = {
                    "emotion": mood.get("mood_label", "unclear"),
                    "genres": mood.get("requested_genres", []),
                    "avoided_genres": mood.get("requested_avoid_genres", []),
                    "request": request_text.strip(),
                }
                st.session_state["recommendations"] = recommendations.to_dict(
                    orient="records"
                )
                st.session_state["recommendation_index"] = 0
                st.session_state["peer_count"] = peer_count
                st.rerun()
        except ValueError as error:
            st.error(str(error))

    request_summary = st.session_state.get("last_request_summary")
    if request_summary and request_summary["request"] == request_text.strip():
        details = [
            f"Emotion estimate: **{request_summary['emotion']}**",
        ]
        if request_summary["genres"]:
            details.append(
                "Requested genre(s): **"
                + ", ".join(request_summary["genres"])
                + "**"
            )
        if request_summary["avoided_genres"]:
            details.append(
                "Avoiding: **"
                + ", ".join(request_summary["avoided_genres"])
                + "**"
            )
        if not request_summary["genres"] and not request_summary["avoided_genres"]:
            details.append(
                "No explicit genre request detected; mood and profile preferences "
                "guide the genre ranking."
            )
        st.info(" · ".join(details))

    render_recommendation_deck(
        profile["profile_id"],
        tmdb_api_key,
        st.session_state.get("peer_count", 0),
    )


def render_recommendation_deck(profile_id: int, tmdb_api_key: str, peer_count: int) -> None:
    recommendations = st.session_state.get("recommendations", [])
    if not recommendations:
        if peer_count:
            st.caption(f"Collaborative signal: {peer_count} similar profile(s).")
        return

    st.divider()
    st.subheader("Your movie matches")
    if peer_count:
        st.caption(
            f"Content-based matches refined with feedback from {peer_count} "
            "taste-similar local profile(s)."
        )
    else:
        st.caption(
            "Ranked from your favorites and this request. Similar-profile learning "
            "will contribute when more local profiles are available."
        )

    index = st.session_state.get("recommendation_index", 0)
    if index >= len(recommendations):
        st.success("You've rated or reviewed all 20 recommendations.")
        feedback = load_feedback(profile_id)
        watchlist_ids = {
            row["item_id"] for row in feedback if row["action"] == "watchlist"
        }
        if watchlist_ids:
            st.markdown("#### Your watchlist")
            content_model = load_content_model()
            titles = content_model.items.set_index("item_id")["title"]
            for item_id in sorted(watchlist_ids):
                if item_id in titles.index:
                    st.write(f"♥ {titles.loc[item_id]}")
        if st.button("Start a new movie search"):
            st.session_state.pop("recommendations", None)
            st.session_state["recommendation_index"] = 0
            st.rerun()
        return

    movie = recommendations[index]
    card_col, details_col = st.columns([1, 2])
    with card_col:
        with st.container(border=True):
            image = poster_for_movie(movie, tmdb_api_key)
            if image:
                st.image(
                    image,
                    caption=movie["title"],
                    use_container_width=True,
                )
            else:
                st.markdown("## 🎬")
                st.caption("Poster unavailable")
    with details_col:
        st.markdown(f"### {movie['title']}")
        if pd.notna(movie.get("release_year")):
            st.caption(str(int(movie["release_year"])))
        st.write(", ".join(parse_genres(movie.get("genres_list", []))) or "Genres unknown")
        st.progress(max(0.0, min(1.0, float(movie.get("final_score", 0.0)))))
        st.caption(f"Movie {index + 1} of {len(recommendations)}")

        with st.form(f"movie_feedback_{profile_id}_{movie['item_id']}"):
            appropriateness = st.slider(
                "How well did this recommendation fit what you asked for?",
                min_value=1,
                max_value=5,
                value=3,
                help="1 = not a fit, 5 = exactly what I wanted",
            )
            watched = st.checkbox("I have already seen this movie")
            movie_rating = None
            if watched:
                movie_rating = st.slider(
                    "Your rating for the movie",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1 = disliked it, 5 = loved it",
                )
            action = st.radio(
                "What would you like to do?",
                ["neutral", "watchlist", "pass"],
                format_func={
                    "neutral": "Just rate it",
                    "watchlist": "♥ Add to watchlist",
                    "pass": "Pass on it",
                }.get,
                horizontal=True,
            )
            if st.form_submit_button(
                "Save feedback and next",
                type="primary",
                use_container_width=True,
            ):
                save_feedback(
                    profile_id,
                    int(movie["item_id"]),
                    appropriateness,
                    movie_rating,
                    action,
                )
                st.session_state["recommendation_index"] = index + 1
                st.rerun()


def main() -> None:
    initialize_store()
    tmdb_api_key = st.sidebar.text_input(
        "TMDb API v3 key",
        value=os.environ.get("TMDB_API_KEY", "").strip(),
        type="password",
        help="Used for movie posters and original-language lookups; Wikipedia is the no-key poster fallback.",
    ).strip()
    groq_api_key = st.sidebar.text_input(
        "Groq API key",
        value=os.environ.get("GROQ_API_KEY", "").strip(),
        type="password",
        help="Used only for Groq-hosted Whisper speech transcription.",
    ).strip()

    st.title("🎬 MoodFlix")
    st.caption("A personal movie finder that learns from your feedback.")
    try:
        content_model = load_content_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run `python download_movielens_32m.py` from the project folder.")
        return

    if not tmdb_api_key:
        st.sidebar.info(
            "Wikipedia will be used to look for poster images. Add a TMDb API v3 "
            "key for more reliable posters and language filtering."
        )
    if (
        st.session_state.get("tmdb_unavailable")
        or st.session_state.get("wikipedia_unavailable")
    ) and st.sidebar.button(
        "Retry poster and TMDb requests"
    ):
        st.session_state.pop("tmdb_unavailable", None)
        st.session_state.pop("tmdb_error_shown", None)
        st.session_state.pop("wikipedia_unavailable", None)
        st.session_state.pop("wikipedia_error_shown", None)
        get_tmdb_info.clear()
        get_tmdb_poster.clear()
        get_wikipedia_poster.clear()
        st.rerun()

    if "active_profile_id" not in st.session_state:
        create_profile_screen(content_model, tmdb_api_key)
    else:
        show_active_profile(content_model, tmdb_api_key, groq_api_key)

    st.caption(
        "Movie catalog: MovieLens 32M (ratings through October 2023). "
        "Posters may come from TMDb or Wikipedia; language metadata is provided "
        "by TMDb. This product is not endorsed or certified by TMDb or Wikipedia."
    )


if __name__ == "__main__":
    main()
