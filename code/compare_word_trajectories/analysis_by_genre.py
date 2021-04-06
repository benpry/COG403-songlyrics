"""
This does the analysis by genre
"""
from cross_correlations import get_best_offset, bootstrap_test
import pandas as pd
import numpy as np

GENRES_FILEPATH = "../../data/processed/genres.csv"
GENRE_DIRECTORY = "../../data/processed/freqs_by_genre"
DATA_PATH = "../../data/processed"
RESULTS_FILE = "../../data/results_by_genre.csv"

if __name__ == "__main__":

    genres = list(pd.read_csv(GENRES_FILEPATH)["genre"].drop_duplicates())

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    rows = []
    for genre in genres:
        df_song = pd.read_csv(f"{GENRE_DIRECTORY}/{genre}.csv")
        df_song = df_song[df_song["year"] >= 1960]

        best_offsets = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)

        mean_offset = np.nanmean(best_offsets)
        p_value = bootstrap_test(mean_offset, best_offsets)

        print(f"{genre} mean offset: {mean_offset}, p={p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")

        rows.append({"genre": genre, "mean_offset": mean_offset, "p": p_value})

    pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)
