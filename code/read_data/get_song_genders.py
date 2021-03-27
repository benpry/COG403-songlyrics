"""
This file gets the gender of the artist of each song and stores the song id with the gender is "songs_with_gender.csv"
"""
import csv

from copy import deepcopy
import pandas as pd
import pickle

ARTIST_GENDERS_PATH = "../../data/processed/artist_genders.csv"
SONGS_PATH = "../../data/processed/songs_with_metadata.p"
SONGS_OUTPUT_PATH = "../../data/processed/songs_with_genders.p"

if __name__ == "__main__":

    # Dictionary {"artist": gender}
    gender_by_artist = {}
    fp = open(ARTIST_GENDERS_PATH, "r", encoding="utf-8")
    reader = csv.reader(fp)
    for row in reader:
        gender_by_artist[row[0]] = row[1]

    # Get song data
    with open(SONGS_PATH, "rb") as fp:
        songs = pickle.load(fp)

    # Generate new list of songs we have gender data for with gender included
    new_songs = []
    for song in songs:
        # Check if the artist is in our gender by artist dictionary
        if song["artist"] in gender_by_artist:
            # If they are, then add this song to list of new songs
            new_song = deepcopy(song)
            new_song["gender"] = gender_by_artist[song["artist"]]
            new_songs.append(new_song)

    # Write new list to file
    with open(SONGS_OUTPUT_PATH, "wb") as fp:
        pickle.dump(new_songs, fp)
