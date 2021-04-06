"""
This file contains the helper functions that compute cross correlations
"""
import numpy as np
import pandas as pd


def time_lag_cross_corrs(s1, s2):
    """
    s1 and s2 are pd.Series objects that we are computing the cross correlations of
    """
    rs = []
    for lag in range(-10, 10):
        r = s1.corr(s2.shift(lag))
        rs.append(r)

    return np.array(rs)


def get_best_offset(df_book, df_song, word):
    """
    This function finds the offset that maximizes the
    """
    book_word = df_book[df_book["word"] == word]["freq"].reset_index()["freq"]
    song_word = df_song[df_song["word"] == word]["freq"].reset_index()["freq"]

    if len(book_word) == 0 or len(song_word) == 0:
        return np.NaN

    rs = time_lag_cross_corrs(song_word, book_word)

    if np.isnan(np.sum(rs)) or np.nanargmax(rs) <= 0:
        return np.NaN

    return np.nanargmax(rs) - 10


def bootstrap_test(val, lst, n=100000):
    """
    Get a p-value using bootstrapping
    """
    n_opposite = 0
    all_subsets = np.random.choice(lst, size=(n, len(lst)), replace=True)
    for i in range(n):
        subset = all_subsets[i, :]
        if np.sign(np.nanmean(subset)) != np.sign(val):
            n_opposite += 1

    return n_opposite / n
