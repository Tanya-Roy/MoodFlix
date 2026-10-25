"""Download MovieLens 32M and create the cleaned files used by MoodFlix."""

import io
import re
import ssl
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "movielensRatings" / "ml-32m"
CLEANED_DIR = DATA_DIR / "cleaned"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
CHUNK_SIZE = 1_000_000


def download_archive(destination: Path) -> None:
    request = Request(DATASET_URL, headers={"User-Agent": "MoodFlix dataset setup"})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(
        request,
        timeout=60,
        context=ssl_context,
    ) as response, destination.open("wb") as output:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded * 100 // total_size
                print(f"\rDownloading MovieLens 32M: {percent:3d}%", end="", flush=True)
        if total_size:
            print()


def clean_title(title: str) -> tuple[str, int | None]:
    match = re.search(r"\s+\((\d{4})\)\s*$", title)
    if not match:
        return title.strip(), None
    return title[: match.start()].strip(), int(match.group(1))


def create_cleaned_files(archive_path: Path) -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            Path(name).name: name
            for name in archive.namelist()
            if Path(name).name in {"ratings.csv", "movies.csv", "links.csv"}
        }
        missing = {"ratings.csv", "movies.csv", "links.csv"}.difference(members)
        if missing:
            raise ValueError(f"MovieLens archive is missing expected files: {sorted(missing)}")

        ratings_path = CLEANED_DIR / "ratings_clean.csv"
        with archive.open(members["ratings.csv"]) as source:
            text_source = io.TextIOWrapper(source, encoding="utf-8", newline="")
            first_chunk = True
            processed = 0
            for chunk in pd.read_csv(
                text_source,
                chunksize=CHUNK_SIZE,
                dtype={
                    "userId": "int32",
                    "movieId": "int32",
                    "rating": "float32",
                    "timestamp": "int32",
                },
            ):
                chunk = chunk.rename(
                    columns={"userId": "user_id", "movieId": "item_id"}
                )[["user_id", "item_id", "rating", "timestamp"]]
                processed += len(chunk)
                chunk.to_csv(
                    ratings_path,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )
                first_chunk = False
                print(f"Processed {processed:,} ratings...", end="\r", flush=True)
            print()

        with archive.open(members["movies.csv"]) as source:
            movies = pd.read_csv(source)
        with archive.open(members["links.csv"]) as source:
            links = pd.read_csv(source, usecols=["movieId", "imdbId"])

    movies = movies.rename(columns={"movieId": "item_id"})
    links = links.rename(columns={"movieId": "item_id"})
    items = movies.merge(links, on="item_id", how="left")
    parsed = items["title"].map(clean_title)
    items["clean_title"] = parsed.map(lambda value: value[0])
    items["release_year"] = parsed.map(lambda value: value[1]).astype("Int64")
    items["tconst"] = items["imdbId"].map(
        lambda value: f"tt{int(value):07d}" if pd.notna(value) else pd.NA
    )
    items["genres_list"] = items["genres"].map(
        lambda value: (
            [genre for genre in value.split("|") if genre and genre != "(no genres listed)"]
            if isinstance(value, str)
            else []
        )
    )
    items["combined_features"] = items["genres_list"].map(
        lambda genres: " ".join(
            genre.lower().replace(" ", "_").replace("'", "") for genre in genres
        )
    )

    items[["item_id", "title", "clean_title", "release_year", "tconst", "genres_list"]].rename(
        columns={"genres_list": "genres"}
    ).to_csv(CLEANED_DIR / "items_clean.csv", index=False)
    items[
        [
            "item_id",
            "title",
            "clean_title",
            "release_year",
            "tconst",
            "genres_list",
            "combined_features",
        ]
    ].to_csv(CLEANED_DIR / "items_with_content_features.csv", index=False)

    print(f"Prepared {len(items):,} movie records.")
    print(f"Prepared ratings at {ratings_path}")
    print(f"Cleaned dataset folder: {CLEANED_DIR}")


def main() -> None:
    if (
        (CLEANED_DIR / "ratings_clean.csv").is_file()
        and (CLEANED_DIR / "items_clean.csv").is_file()
        and (CLEANED_DIR / "items_with_content_features.csv").is_file()
    ):
        print(f"Cleaned MovieLens 32M files already exist at {CLEANED_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="ml-32m-", suffix=".zip", delete=False
    ) as temporary:
        archive_path = Path(temporary.name)

    try:
        print("Downloading the official GroupLens MovieLens 32M archive...")
        download_archive(archive_path)
        print("Extracting and preparing the ratings and movie metadata...")
        create_cleaned_files(archive_path)
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
