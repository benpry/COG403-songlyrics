"""
Computes a separate frequency dataframe for a sample of 100 songs by each age group.
"""
import pickle
import pandas as pd
from compute_aggregate_song_frequencies import compute_frequency_dataframe
from random import sample

SONGS_FILEPATH = "../../data/processed/songs_with_ages.p"
OUTPUT_DIRECTORY = "../../data/processed/freqs_by_age"

if __name__ == "__main__":

    with open(SONGS_FILEPATH, "rb") as fp:
        all_songs = pickle.load(fp)

    def rule(group, song):
        if group == 'under20':
            return int(song["age"]) < 20
        elif group == '20to30':
            return 20 <= int(song["age"]) and int(song["age"]) < 30
        elif group == '30to40':
            return 30 <= int(song["age"]) and int(song["age"]) < 40 
        elif group == '40to50':
            return 40 <= int(song["age"]) and int(song["age"]) < 50
        else:
            return 50 <= int(song["age"])

    age_groups = ['under20', '20to30', '30to40', '40to50', 'over50']
    for group in age_groups:
        songs = [song for song in all_songs if rule(group, song)]
        songs_sample = sample(songs, 100)
        df = compute_frequency_dataframe(songs_sample)
        df.to_csv(f"{OUTPUT_DIRECTORY}/{group}.csv")


    
