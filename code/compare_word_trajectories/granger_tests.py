"""
This file contains utility functions for granger causality testing
"""
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import InfeasibleTestError


def run_granger_test(df_book, df_song, word):

    book_word = df_book[df_book["word"] == word]["freq"].reset_index()["freq"]
    song_word = df_song[df_song["word"] == word]["freq"].reset_index()["freq"]

    try:
        forward_result = grangercausalitytests(np.array([song_word, book_word]).transpose(), maxlag=10, addconst=True, verbose=False)
        backward_result = grangercausalitytests(np.array([book_word, song_word]).transpose(), maxlag=10, addconst=True, verbose=False)
    except ValueError:
        return np.NaN
    except InfeasibleTestError:
        return np.NaN

    best_lag = None
    best_F = -1
    for forward_lag in forward_result:
        if forward_result[forward_lag][0]["ssr_ftest"][1] > best_F:
            best_lag = - forward_lag
            best_F = forward_result[forward_lag][0]["ssr_ftest"][1]
    for backward_lag in backward_result:
        if backward_result[backward_lag][0]["ssr_ftest"][1] > best_F:
            best_lag = backward_lag
            best_F = backward_result[backward_lag][0]["ssr_ftest"][1]

    return best_lag
