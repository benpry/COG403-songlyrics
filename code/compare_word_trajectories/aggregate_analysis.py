"""
This file computes the cross correlations and ahead-ness of words in music
"""
from cross_correlations import get_best_offset, bootstrap_test
import pandas as pd
import numpy as np
import pickle
from granger_tests import run_granger_test

DATA_PATH = "../../data/processed"
RESULTS_PATH = "../../data/aggregate_results.csv"

if __name__ == "__main__":

    df_song = pd.read_csv(f"{DATA_PATH}/song_frequencies.csv")
    df_song = df_song[df_song["year"] >= 1980]

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1980) & (df_book["year"] <= 2011)]

    rows = []
    best_offsets = []
    granger_results = []
    for word in df_book["word"].drop_duplicates():
        best_offset = get_best_offset(df_book, df_song, word)
        best_offsets.append(best_offset)
        granger_result = run_granger_test(df_book, df_song, word)
        granger_results.append(granger_result)
        rows.append({"word": word, "offset": best_offset, "granger_lag": granger_result})

    mean_offset = np.nanmean(best_offsets)
    p_value = bootstrap_test(mean_offset, best_offsets)

    df_offsets = pd.DataFrame(rows)
    df_offsets.to_csv(RESULTS_PATH, index=False)

    print(f"mean offset: {mean_offset}, p={p_value}")
    print(f"median offset: {np.nanmedian(best_offsets)}")

    mean_granger_lag = np.nanmean(granger_results)
    granger_p = bootstrap_test(mean_granger_lag, granger_results)

    print(f"mean granger lag: {mean_granger_lag}, p={granger_p}")
    print(f"median offset: {np.nanmedian(granger_results)}")
