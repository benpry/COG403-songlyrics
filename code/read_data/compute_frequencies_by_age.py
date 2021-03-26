"""
Computes a separate frequency dataframe for each gender
"""
import pickle
import pandas as pd
from compute_aggregate_song_frequencies import compute_frequency_dataframe

SONGS_FILEPATH = "../../data/processed/songs_with_ages.p"
OUTPUT_DIRECTORY = "../../data/processed/freqs_by_age"

if __name__ == "__main__":

    with open(SONGS_FILEPATH, "rb") as fp:
        all_songs = pickle.load(fp)

    age_under20_songs = [song for song in all_songs if int(song["age"]) < 20 ]
    df_under20 = compute_frequency_dataframe(age_under20_songs)
    df_under20.to_csv(f"{OUTPUT_DIRECTORY}/under20.csv")

    age_20to30_songs = [song for song in all_songs if ( 20 <= int(song["age"]) and int(song["age"]) < 30 )]
    df_20to30 = compute_frequency_dataframe(age_20to30_songs)
    df_20to30.to_csv(f"{OUTPUT_DIRECTORY}/20to30.csv")

    age_30to40_songs = [song for song in all_songs if ( 30 <= int(song["age"]) and int(song["age"]) < 40 )]
    df_30to40 = compute_frequency_dataframe(age_30to40_songs)
    df_30to40.to_csv(f"{OUTPUT_DIRECTORY}/30to40.csv")

    age_40to50_songs = [song for song in all_songs if ( 40 <= int(song["age"]) and int(song["age"]) < 50 )]
    df_40to50 = compute_frequency_dataframe(age_40to50_songs)
    df_40to50.to_csv(f"{OUTPUT_DIRECTORY}/40to50.csv")

    age_over50_songs = [song for song in all_songs if 50 <= int(song["age"])]
    df_over50 = compute_frequency_dataframe(age_over50_songs)
    df_over50.to_csv(f"{OUTPUT_DIRECTORY}/over50.csv")


    # TODO: Refactor
    # age_groups = ["under20", "20to30", "30to40", "40to50", "over50"]
    # for age_group in age_groups:
    #     age_group_songs = [song for song in all_songs if lookup(song["age"]) == age_group]
    #     df_group = compute_frequency_dataframe(age_group_songs)
    #     df_group.to_csv(f"{OUTPUT_DIRECTORY}/{age_group}.csv")

    