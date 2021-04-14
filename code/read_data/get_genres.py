"""
This file parses the genre information from the genre file
"""
from copy import deepcopy
import pandas as pd
import pickle

GENRE_FILEPATH = "../../data/genre_tags.cls"
SONGS_FILEPATH = "../../data/processed/songs_with_metadata.p"
SONGS_OUTPUT_FILEPATH = "../../data/processed/songs_with_genres.p"

if __name__ == "__main__":

    with open(GENRE_FILEPATH, "r") as fp:
        lines = fp.readlines()

    genres_by_id = {}
    for line in lines:
        elements = line.replace("\n", "").split("\t")
        if len(elements) <= 1:
            continue

        track_id = elements[0]
        genre = elements[1]

        genres_by_id[track_id] = genre

    with open(SONGS_FILEPATH, "rb") as fp:
        songs = pickle.load(fp)

    new_songs = []
    for song in songs:
        new_song = deepcopy(song)
        if song["id"] not in genres_by_id:
            continue
        new_song["genre"] = genres_by_id[song["id"]]
        new_songs.append(new_song)

    with open(SONGS_OUTPUT_FILEPATH, "wb") as fp:
        pickle.dump(new_songs, fp)
