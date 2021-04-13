"""
Computes a separate frequency dataframe for a sample of 3000 songs by each gender.
"""
import pickle
import pandas as pd
from compute_aggregate_song_frequencies import compute_frequency_dataframe
from random import sample

SONGS_FILEPATH = "../../data/processed/songs_with_genders.p"
OUTPUT_DIRECTORY = "../../data/processed/freqs_by_gender"

if __name__ == "__main__":

    with open(SONGS_FILEPATH, "rb") as fp:
        all_songs = pickle.load(fp)

    genders = ['female', 'male']

    for gender in genders:
        songs = [song for song in all_songs if song["gender"] == gender]
        songs_sample = sample(songs, 3000)
        df_gender = compute_frequency_dataframe(songs_sample)
        df_gender.to_csv(f"{OUTPUT_DIRECTORY}/{gender}.csv")