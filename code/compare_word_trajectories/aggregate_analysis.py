"""
This file computes the cross correlations and ahead-ness of words in music
"""
from cross_correlations import time_lag_cross_corrs
import pandas as pd
import numpy as np
import pickle

DATA_PATH = "../../data/processed"


def get_best_offset(df_book, df_song, word):
    """
    This function finds the offset that maximizes the
    """
    book_word = df_book[df_book["word"] == word]["freq"].reset_index()["freq"]
    song_word = df_song[df_song["word"] == word]["freq"].reset_index()["freq"]

    rs = time_lag_cross_corrs(book_word, song_word)
    if np.isnan(np.sum(rs)):
        return 0

    return np.nanargmax(rs) - len(book_word) // 2


def bootstrap_test(val, lst, n=100000):
    n_opposite = 0
    all_subsets = np.random.choice(lst, size=(n, len(lst)), replace=True)
    for i in range(n):
        subset = all_subsets[i, :]
        if np.sign(np.mean(subset)) != np.sign(val):
            n_opposite += 1

    return n_opposite / n


if __name__ == "__main__":

    df_song = pd.read_csv(f"{DATA_PATH}/song_frequencies.csv")
    df_song = df_song[df_song["year"] >= 1960]

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    best_offsets = []
    for word in df_book["word"].drop_duplicates():
        best_offset = get_best_offset(df_book, df_song, word)
        best_offsets.append(best_offset)

    mean_offset = np.mean(best_offsets)
    p_value = bootstrap_test(mean_offset, best_offsets)

    print(f"mean offset: {mean_offset}, p={p_value}")
