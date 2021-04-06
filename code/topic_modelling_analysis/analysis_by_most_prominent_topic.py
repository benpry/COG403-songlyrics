"""
This file gets the best offset for each topic and runs bootstrap tests
"""
import sys
sys.path.append("../compare_word_trajectories")
from cross_correlations import get_best_offset, bootstrap_test
import pandas as pd
import numpy as np
import pickle

N_TOPICS = 30
DATA_PATH = "../../data/processed"
SONGS_FILEPATH = "../../data/processed/songs_with_top_topics.p"
TOPIC_DIRECTORY = "../../data/processed/freqs_by_topic"
RESULTS_FILEPATH = "../../data/topic_modelling_results.csv"

if __name__ == "__main__":

    with open(SONGS_FILEPATH, "rb") as fp:
        songs = pickle.load(fp)

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    rows = []
    # separate songs by top topic
    for topic in range(N_TOPICS):
        df_song = pd.read_csv(f"{TOPIC_DIRECTORY}/topic_{topic}.csv")
        df_song = df_song[df_song["year"] >= 1960]

        best_offsets = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)

        mean_offset = np.nanmean(best_offsets)
        p_value = bootstrap_test(mean_offset, best_offsets)

        print(f"topic {topic} mean offset: {mean_offset}, p={p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")
        rows.append({"topic": topic, "offset": mean_offset, "p": p_value})

    pd.DataFrame(rows).to_csv(RESULTS_FILEPATH, index=False)
