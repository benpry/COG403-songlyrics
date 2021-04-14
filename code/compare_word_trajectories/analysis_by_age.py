"""
This does the analysis by age
"""
from cross_correlations import get_best_offset, bootstrap_test
from granger_tests import run_granger_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AGE_DIRECTORY = "../../data/processed/freqs_by_age"
DATA_PATH = "../../data/processed"
RESULTS_FILE = "../../data/results_by_age.csv"
PLOT_PATH = "../../figures"

if __name__ == "__main__":

    ages = ["under20", "20to30", "30to40", "40to50", "over50"]

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1980) & (df_book["year"] <= 2011)]

    # Get mean offsets and granger lags for each age group
    rows = []
    for age in ages:
        df_song = pd.read_csv(f"{AGE_DIRECTORY}/{age}.csv")
        df_song = df_song[df_song["year"] >= 1980]

        best_offsets = []
        granger_lags = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)
            granger_lag = run_granger_test(df_book, df_song, word)
            granger_lags.append(granger_lag)

        mean_offset = np.nanmean(best_offsets)
        p_value = bootstrap_test(mean_offset, best_offsets)

        print(f"{age} mean offset: {mean_offset}, p={p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")

        mean_granger_lag = np.nanmean(granger_lags)
        granger_p_val = bootstrap_test(mean_granger_lag, granger_lags)

        print(f"{age} granger lag: {mean_granger_lag}, p={granger_p_val}")
        print(f"median offset: {np.nanmedian(granger_lags)}")

        rows.append({"age": age, "mean_offset": mean_offset, "offset_p": p_value, "mean_granger_lag": mean_granger_lag,
                     "granger_p": granger_p_val})

    # Store results
    pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)

    # Plot results
    mean_offsets = [row['mean_offset'] for row in rows]
    mean_granger_lags = [row['mean_granger_lag'] for row in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 18))
    ax1.bar(ages, mean_offsets)
    ax1.set_xlabel('Age Group')
    ax1.set_xticks([i for i in range(len(ages))])
    ax1.set_xticklabels(ages)
    ax1.set_ylabel('Mean Offset')
    ax1.set_title('Mean Offset by Age Group')
    ax2.bar(ages, mean_granger_lags)
    ax2.set_xlabel('Age Group')
    ax2.set_xticks([i for i in range(len(ages))])
    ax2.set_xticklabels(ages)
    ax2.set_ylabel('Mean Granger Lag')
    ax2.set_title('Mean Granger Lag by Age Group')
    plt.savefig(PLOT_PATH + '/age_analysis_plot.png')


