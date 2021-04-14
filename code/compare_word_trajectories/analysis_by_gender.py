"""
This does the analysis by gender
"""
from cross_correlations import get_best_offset, bootstrap_test
from granger_tests import run_granger_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

GENDER_DIRECTORY = "../../data/processed/freqs_by_gender"
DATA_PATH = "../../data/processed"
RESULTS_FILE = "../../data/results_by_gender.csv"
PLOT_PATH = "../../figures"

if __name__ == "__main__":

    genders = ['female', 'male']

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1980) & (df_book["year"] <= 2011)]

    rows = []
    for gender in genders:
        df_song = pd.read_csv(f"{GENDER_DIRECTORY}/{gender}.csv")
        df_song = df_song[df_song["year"] >= 1980]

        best_offsets = []
        granger_lags = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)
            granger_lag = run_granger_test(df_book, df_song, word)
            granger_lags.append(granger_lag)

        # Get offset mean, median and p value
        mean_offset = np.nanmean(best_offsets)
        offset_p_value = bootstrap_test(mean_offset, best_offsets)
        print(f"{gender} mean offset: {mean_offset}, p={offset_p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")

        # Get offset standard error
        best_offsets = [value for value in best_offsets if not math.isnan(value)]
        offset_error = stats.sem(best_offsets)
        print(f"offset standard error: {offset_error}")

        # Get Granger lag mean, median and p value
        mean_granger_lag = np.nanmean(granger_lags)
        granger_p_val = bootstrap_test(mean_granger_lag, granger_lags)
        print(f"{gender} granger lag: {mean_granger_lag}, p={granger_p_val}")
        print(f"median offset: {np.nanmedian(granger_lags)}")

        # Get Granger lag standard error
        granger_lags = [value for value in granger_lags if not math.isnan(value)]
        granger_error = stats.sem(granger_lags)
        print(f"granger lag standard error: {granger_error}")

        rows.append({"gender": gender, "mean_offset": mean_offset, "offset_error": offset_error, "offset_p": offset_p_value,
            "mean_granger_lag": mean_granger_lag, "granger_error": granger_error, "granger_p": granger_p_val})
    
    # Store results
    pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)

    # Plot results
    labels = genders
    mean_offsets = [row['mean_offset'] for row in rows]
    offset_errors = [row['offset_error'] for row in rows]
    offset_p_values = [row['offset_p'] for row in rows]
    mean_granger_lags = [row['mean_granger_lag'] for row in rows]
    granger_errors = [row['granger_error'] for row in rows]
    granger_p_values = [row['granger_p'] for row in rows]
    p_value_text = [str(genders[i]) + ": " + "Offset - " + str(offset_p_values[i]) + ", Granger Lag - " + str(granger_p_values[i]) for i in range(len(genders))]
    
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, mean_offsets, width, yerr=offset_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='Offsets')
    rects2 = ax.bar(x + width/2, mean_granger_lags, width, yerr=granger_errors, align='center', alpha=0.5, ecolor='black', capsize=10, label='Granger Lags')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Offset and Mean Granger Lag by Gender')
    ax.set_xlabel('P values: \n' + ',\n'.join(p_value_text))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(PLOT_PATH + '/gender_analysis_plot.png')
