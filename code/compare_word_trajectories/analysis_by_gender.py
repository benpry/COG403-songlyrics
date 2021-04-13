"""
This does the analysis by gender
"""
from cross_correlations import get_best_offset, bootstrap_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GENDER_DIRECTORY = "../../data/processed/freqs_by_gender"
DATA_PATH = "../../data/processed"
RESULTS_PATH = '../../results'

if __name__ == "__main__":

    genders = ['female', 'male']
    mean_offsets = []

    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    for gender in genders:
        df_song = pd.read_csv(f"{GENDER_DIRECTORY}/{gender}.csv")
        df_song = df_song[df_song["year"] >= 1960]

        best_offsets = []
        for word in df_book["word"].drop_duplicates():
            best_offset = get_best_offset(df_book, df_song, word)
            best_offsets.append(best_offset)

        mean_offset = np.nanmean(best_offsets)
        p_value = bootstrap_test(mean_offset, best_offsets)

        print(f"{gender} mean offset: {mean_offset}, p={p_value}")
        print(f"median offset: {np.nanmedian(best_offsets)}")

        mean_offsets.append(mean_offset)
    
    # Plot
    fig, ax = plt.subplots()
    fig.suptitle('Gender Analysis Results Plotted', y=0.92)
    ax = plt.scatter(genders, mean_offsets)
    plt.xlabel('Genders')
    plt.ylabel('Mean Offset')
    plt.savefig(RESULTS_PATH + "/gender_analysis_plot.png")
