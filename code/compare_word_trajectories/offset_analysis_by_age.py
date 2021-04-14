"""
This does the analysis by age
"""
from cross_correlations import get_best_offset, bootstrap_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AGE_DIRECTORY = "../../data/processed/freqs_by_age"
DATA_PATH = "../../data/processed"
RESULTS_PATH = '../../results'

if __name__ == "__main__":

    ages = ["under20", "20to30", "30to40", "40to50", "over50"]
    mean_offsets = []

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    for age in ages:
        df_song = pd.read_csv(f"{AGE_DIRECTORY}/{age}.csv")
        df_song = df_song[df_song["year"] >= 1960]

        best_offsets = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)

        mean_offset = np.nanmean(best_offsets)
        p_value = bootstrap_test(mean_offset, best_offsets)

        print(f"{age} mean offset: {mean_offset}, p={p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")

        mean_offsets.append(mean_offset)
    
    # Plot
    fig, ax = plt.subplots()
    fig.suptitle('Mean Offsets by Age Group Results Plotted', y=0.92)
    ax = plt.bar(ages, mean_offsets)
    plt.xlabel('Age Groups')
    plt.ylabel('Mean Offset')
    plt.savefig(RESULTS_PATH + "/age_offset_analysis_plot.png")