"""
This file goes through 'songs.p' and adds the metadata from each song to the recording
"""
from copy import deepcopy
import pandas as pd
import numpy as np
import pickle

DATA_PATH = "../../data/processed"

if __name__ == "__main__":

    # load the songs
    with open(f"{DATA_PATH}/songs.p", 'rb') as fp:
        songs = pickle.load(fp)

    # read the metadata
    df_metadata = pd.read_csv(f"{DATA_PATH}/all_metadata.csv", index_col=0).dropna().drop_duplicates()

    # add metadata to the songs
    new_songs = []
    for song in songs:
        # make sure we have metadata for this song
        if song["id"] not in df_metadata.index:
            continue

        row = df_metadata.loc[song["id"]]

        new_song = deepcopy(song)
        new_song["year"] = int(row["year"])
        new_song["artist"] = row["artist"]
        new_song["title"] = row["title"]

        new_songs.append(new_song)

    print(f"total number of songs: {len(songs)}")
    print(f"number of songs with metadata: {len(new_songs)}")

    with open(f"{DATA_PATH}/songs_with_metadata.p", "wb") as fp:
        pickle.dump(new_songs, fp)
