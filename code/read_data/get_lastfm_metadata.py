"""
This file saves the subset of the last.fm dataset that corresponds to the songs in the MusixMatch dataset
"""
import pickle
import sys
import json
import pandas as pd
# import requests
from tqdm import tqdm

SONGS_PATH = "../../data/processed/songs.p"
LASTFM_PATH = "../../data/lastfm_train"
METADATA_PATH = "../../data/processed/lastfm_metadata.csv"
FILEPATH_PATH = "../../data/filepaths.csv"

if __name__ == "__main__":

    with open(SONGS_PATH, "rb") as fp:
        songs = pickle.load(fp)

    ids = set([x["id"] for x in songs])

    full_paths = list(pd.read_csv(FILEPATH_PATH)["0"])

    all_metadata = []
    for full_path in tqdm(full_paths):
        with open(full_path, "r") as fp:
            data = json.load(fp)
        # read the artist and timestamp
        artist, title = data["artist"], data["title"]
        mxm_id = full_path.split("/")[-1].split(".")[0]

        # mb_metadata = requests.get(f'http://musicbrainz.org/ws/2/recording?query=artist:{artist},recording:{title}&fmt=json&limit=1').json()
        # if len(mb_metadata["recordings"]) > 0 and "first-release-date" in mb_metadata["recordings"][0]:
        # release_date = mb_metadata["recordings"][0]["first-release-date"]
        all_metadata.append({"id": mxm_id, "artist": artist, "title": title})  # "release date": release_date})

    pd.DataFrame(all_metadata).to_csv(METADATA_PATH, index=False)
