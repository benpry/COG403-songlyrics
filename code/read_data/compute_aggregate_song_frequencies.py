"""
This file computes the aggregate frequencies of word in songs.
"""
import pickle
import pandas as pd
from collections import defaultdict

DATA_PATH = "../../data/processed"
ALL_WORDS_PATH = "../../data/processed/allwords.p"

with open(ALL_WORDS_PATH, "rb") as fp:
    all_words = pickle.load(fp)


def default_dict_int():
    return defaultdict(int)


def compute_frequency_dataframe(songs):
    # defaultdict of defaultdicts of integers
    yearly_word_freqs = defaultdict(default_dict_int)
    for song in songs:
        year = song["year"]
        for word in song["word_freqs"].keys():
            yearly_word_freqs[year][word] += song["word_freqs"][word]

    rows = []
    for year in range(1900, 2012):
        yearly_total = sum(yearly_word_freqs[year].values())

        if yearly_total == 0:
            for word in all_words:
                rows.append({"year": year, "word": word, "freq": 0})
        else:
            for word in all_words:
                rows.append({"year": year, "word": word, "freq": yearly_word_freqs[year][word] / yearly_total})

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":

    with open(f"{DATA_PATH}/songs_with_metadata.p", "rb") as fp:
        songs = pickle.load(fp)

    with open(ALL_WORDS_PATH, "rb") as fp:
        all_words = pickle.load(fp)

    df = compute_frequency_dataframe(songs)

    df.to_csv(f"{DATA_PATH}/song_frequencies.csv", index=False)
