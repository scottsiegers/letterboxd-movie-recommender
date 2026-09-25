import io
import zipfile
from pathlib import Path

import pandas as pd


EXPECTED_FILES = {
    "ratings.csv": "ratings",
    "reviews.csv": "reviews",
    "watched.csv": "watched",
    "diary.csv": "diary",
    "watchlist.csv": "watchlist",
}


def _empty_data():
    return {
        "ratings": None,
        "reviews": None,
        "watched": None,
        "diary": None,
        "watchlist": None,
    }


def _normalize_filename(filename):
    """
    Return just the file name, lowercased.

    Handles:
      ratings.csv
      export/ratings.csv
      some/folder/Ratings.csv
    """
    return Path(filename).name.lower()


def _read_csv_file(file_obj):
    """
    Read a CSV from a Streamlit UploadedFile or other file-like object.
    """
    file_obj.seek(0)
    return pd.read_csv(file_obj)


def _parse_csv_uploads(uploaded_files):
    """
    Parse individually uploaded CSV files or files selected via
    Streamlit directory upload.
    """
    data = _empty_data()

    for uploaded_file in uploaded_files:
        filename = _normalize_filename(uploaded_file.name)

        if filename not in EXPECTED_FILES:
            continue

        key = EXPECTED_FILES[filename]

        try:
            data[key] = _read_csv_file(uploaded_file)
        except Exception as exc:
            raise ValueError(
                f"Could not read {uploaded_file.name}: {exc}"
            ) from exc

    return data


def _parse_zip_upload(zip_file):
    """
    Parse CSV files directly from a Letterboxd ZIP export.
    """
    data = _empty_data()

    zip_file.seek(0)

    try:
        with zipfile.ZipFile(zip_file) as archive:

            for member_name in archive.namelist():
                filename = _normalize_filename(member_name)

                if filename not in EXPECTED_FILES:
                    continue

                key = EXPECTED_FILES[filename]

                with archive.open(member_name) as csv_file:
                    csv_bytes = csv_file.read()

                try:
                    data[key] = pd.read_csv(
                        io.BytesIO(csv_bytes)
                    )

                except Exception as exc:
                    raise ValueError(
                        f"Could not read {member_name} from ZIP: {exc}"
                    ) from exc

    except zipfile.BadZipFile as exc:
        raise ValueError(
            "The uploaded ZIP file could not be opened."
        ) from exc

    return data


def merge_letterboxd_data(base_data, new_data):
    """
    Merge two parsed Letterboxd datasets.

    New non-empty datasets replace existing ones.
    """
    merged = base_data.copy()

    for key, dataframe in new_data.items():
        if dataframe is not None:
            merged[key] = dataframe

    return merged


def parse_letterboxd_uploads(
    zip_file=None,
    csv_files=None,
    folder_files=None,
):
    """
    Accept any combination of:

      - ZIP export
      - manually selected CSV files
      - directory/folder upload

    Returns a dictionary containing parsed DataFrames.
    """

    data = _empty_data()

    if zip_file is not None:
        zip_data = _parse_zip_upload(zip_file)

        data = merge_letterboxd_data(
            data,
            zip_data,
        )

    if folder_files:
        folder_data = _parse_csv_uploads(folder_files)

        data = merge_letterboxd_data(
            data,
            folder_data,
        )

    if csv_files:
        csv_data = _parse_csv_uploads(csv_files)

        data = merge_letterboxd_data(
            data,
            csv_data,
        )

    return data


def build_user_movies(data):
    """
    Build one clean movie dataframe from ratings and reviews.
    """

    ratings = data.get("ratings")
    reviews = data.get("reviews")

    if ratings is None:
        raise ValueError(
            "ratings.csv was not found. "
            "Please upload a Letterboxd export containing ratings.csv."
        )

    required_rating_columns = {"Name", "Year", "Rating"}

    missing = required_rating_columns - set(ratings.columns)

    if missing:
        raise ValueError(
            "ratings.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    rating_cols = [
        col
        for col in [
            "Date",
            "Name",
            "Year",
            "Letterboxd URI",
            "Rating",
        ]
        if col in ratings.columns
    ]

    user_movies = ratings[rating_cols].copy()

    if reviews is not None:

        required_review_columns = {
            "Name",
            "Year",
            "Review",
        }

        if required_review_columns.issubset(
            reviews.columns
        ):

            reviews_clean = reviews[
                ["Name", "Year", "Review"]
            ].copy()

            reviews_clean = (
                reviews_clean
                .drop_duplicates(
                    subset=["Name", "Year"],
                    keep="last",
                )
            )

            user_movies = user_movies.merge(
                reviews_clean,
                on=["Name", "Year"],
                how="left",
            )

    user_movies = (
        user_movies
        .drop_duplicates(
            subset=["Name", "Year"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return user_movies


def get_upload_summary(data):
    """
    Return record counts for each supported Letterboxd file.
    """

    summary = {}

    for key, dataframe in data.items():
        summary[key] = (
            0
            if dataframe is None
            else len(dataframe)
        )

    return summary


def get_detected_files(data):
    """
    Return a list of detected Letterboxd datasets.
    """

    return [
        key
        for key, dataframe in data.items()
        if dataframe is not None
    ]