"""
Computes a separate frequency dataframe for each genre of music
"""
import pickle
import pandas as pd
from compute_aggregate_song_frequencies import compute_frequency_dataframe

SONGS_FILEPATH = "../../data/processed/songs_with_genres.p"
GENRES_FILEPATH = "../../data/processed/genres.csv"
OUTPUT_DIRECTORY = "../../data/processed/freqs_by_genre"

if __name__ == "__main__":

    with open(SONGS_FILEPATH, "rb") as fp:
        all_songs = pickle.load(fp)

    genres = list(pd.read_csv(GENRES_FILEPATH)["genre"].drop_duplicates())

    for genre in genres:
        genre_songs = [song for song in all_songs if song["genre"] == genre]

        df_genre = compute_frequency_dataframe(genre_songs)
        df_genre.to_csv(f"{OUTPUT_DIRECTORY}/{genre}.csv")
