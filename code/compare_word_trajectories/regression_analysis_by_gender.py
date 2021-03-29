"""
This does the regression analysis by gender
"""

import pandas as pd
from sklearn import linear_model

GENDER_DIRECTORY = "../../data/processed/freqs_by_gender"
DATA_PATH = "../../data/processed"

if __name__ == "__main__":

    # Get all word frequencies per year from books
    df_book = pd.read_csv(f"{DATA_PATH}/book_frequencies.csv")
    df_book = df_book.drop(["Corpus"], axis=1)
    df_book.columns = ["", "year", "word", "freq"]
    df_book = df_book.drop([""], axis=1)
    df_book = df_book[(df_book["year"] >= 1960) & (df_book["year"] <= 2011)]

    # Get all word frequencies per year from lyrics by female artists
    df_song_female = pd.read_csv(f"{GENDER_DIRECTORY}/female.csv")
    df_song_female = df_song_female[df_song_female["year"] >= 1960]
    df_song_female = df_song_female.drop(["Unnamed: 0"], axis=1)

    # Get all word frequencies per year from lyrics by male artists
    df_song_male = pd.read_csv(f"{GENDER_DIRECTORY}/male.csv")
    df_song_male = df_song_male[df_song_male["year"] >= 1960]
    df_song_male = df_song_male.drop(["Unnamed: 0"], axis=1)

    # Join to compare freqs in female and male
    # This gives a df with columns [year, word, freq in female, freq in male]
    df_joined = pd.merge(df_song_female, df_song_male, how='inner', on=['year', 'word'])
    # Add freq in books
    # This gives a df with columns [year, word, freq in female, freq in male. freq in books]
    df_joined = pd.merge(df_joined, df_book, how='inner', on=['year', 'word'])
    df_joined.columns = ["year", "word", "freq_female", "freq_male", "freq_books"]
    # Add differene in freqs
    df_joined["diff_female"] = df_joined["freq_female"] - df_joined["freq_books"]
    df_joined["diff_male"] = df_joined["freq_male"] - df_joined["freq_books"]

    print(df_joined.head())

    # Average across all words per year
    df_new = df_joined[["year", "diff_female", "diff_male"]].copy()
    df_new = df_new.groupby("year").mean()
    print(df_new.head())


