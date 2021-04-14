"""
This file does the regression analysis by gender.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

GENDER_DIRECTORY = "../../data/processed/freqs_by_gender"
OFFSETS_PATH = "../../data/aggregate_best_offsets.csv"
RESULTS_PATH = '../../results'

if __name__ == "__main__":

    # Get all word frequencies per year from lyrics by male artists
    df_song_male = pd.read_csv(f"{GENDER_DIRECTORY}/male.csv")
    df_song_male = df_song_male[df_song_male["year"] >= 1980]
    df_song_male = df_song_male.drop(["Unnamed: 0"], axis=1)

    # Get all word frequencies per year from lyrics by female artists
    df_song_female = pd.read_csv(f"{GENDER_DIRECTORY}/female.csv")
    df_song_female = df_song_female[df_song_female["year"] >= 1980]
    df_song_female = df_song_female.drop(["Unnamed: 0"], axis=1)

    # Join 
    df_joined = pd.merge(df_song_male, df_song_female, how='inner', on=['year', 'word'])
    df_joined = df_joined.rename(columns={"freq_x": "freq_male", "freq_y": "freq_female"}, errors="raise")

    # Average across years for each word
    df_joined = df_joined.groupby('word').mean().reset_index()
    df_joined = df_joined.drop(["year"], axis=1)

    # Add offsets
    df_offsets = pd.read_csv(OFFSETS_PATH, sep=',')
    df_offsets = df_offsets.replace(np.nan, 0)
    df_joined = pd.merge(df_joined, df_offsets, how='inner', on=['word'])

    # Level-level Regression
    X = df_joined[['freq_male','freq_female']]
    y = df_joined['offset']
    X = sm.add_constant(X)
    levlev_model = sm.OLS(y, X).fit()

    # Level-log Regression
    df_joined['log_freq_male'] = np.where(df_joined['freq_male'] > 0, np.log2(df_joined['freq_male']), 0)
    df_joined['log_freq_female'] = np.where(df_joined['freq_female'] > 0, np.log2(df_joined['freq_female']), 0)
    X = df_joined[['log_freq_male','log_freq_female']]
    y = df_joined['offset']
    X = sm.add_constant(X)
    levlog_model = sm.OLS(y, X).fit()

    # Log-log Regression
    np.seterr(divide = 'ignore')
    df_joined['log_offset'] = np.where(df_joined['offset'] > 0, np.log2(df_joined['offset']), 0)
    X = df_joined[['log_freq_male','log_freq_female']]
    y = df_joined['log_offset']
    X = sm.add_constant(X) 
    loglog_model = sm.OLS(y, X).fit()
    print(loglog_model.summary())

    # Summaries
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle('Gender Regression Results', y=0.92)
    ax1.text(0.01, 0.05, str(levlev_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax1.axis('off')
    ax2.text(0.01, 0.05, str(levlog_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax2.axis('off')
    ax3.text(0.01, 0.05, str(loglog_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "/gender_regression_summaries.png")

