"""
This file saves the subset of the last.fm dataset that corresponds to the songs in the MusixMatch dataset
"""
import pickle
import sys
import pandas as pd
import requests
from tqdm import tqdm

SONGS_PATH = "../../data/processed/songs.p"
LASTFM_PATH = "../../data/lastfm_train"
METADATA_PATH = "../../data/processed/lastfm_metadata.csv"
FILEPATH_PATH = "../../data/filepaths.csv"

if __name__ == "__main__":

    subset_idx = int(sys.argv[1])

    dates_path = f"../../data/processed/dates_{subset_idx}.csv"

    with open(SONGS_PATH, "rb") as fp:
        songs = pickle.load(fp)

    ids = set([x["id"] for x in songs])

    df_metadata = pd.read_csv(METADATA_PATH)

    start_idx, finish_idx = int((subset_idx / 4) * len(df_metadata)), int((subset_idx + 1) / 4 * len(df_metadata))
    df_metadata = df_metadata.loc[start_idx:finish_idx]

    all_dates = []
    for index, row in tqdm(list(df_metadata.iterrows())[:10]):
        # read the artist and timestamp
        artist, title = row["artist"], row["title"]

        mb_metadata = requests.get(f'http://musicbrainz.org/ws/2/recording?query=artist:{artist},recording:{title}&fmt=json&limit=1').json()
        if len(mb_metadata["recordings"]) > 0 and "first-release-date" in mb_metadata["recordings"][0]:
            release_date = mb_metadata["recordings"][0]["first-release-date"]
            all_dates.append({"id": row["id"], "date": release_date})

    pd.DataFrame(all_dates).to_csv(dates_path, index=False)
