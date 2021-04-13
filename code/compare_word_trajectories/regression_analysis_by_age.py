"""
This does the regression analysis by age
"""

import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.api as sm
import matplotlib.pyplot as plt

AGE_DIRECTORY = "../../data/processed/freqs_by_age"
DATA_PATH = "../../data/processed"
OFFSETS_PATH = "../../data/aggregate_best_offsets.csv"
RESULTS_PATH = "../../results"

if __name__ == "__main__":

    # Get all word frequencies per year from lyrics by age groups
    age_groups = ['under20', '20to30', '30to40', '40to50', 'over50']
    freq_dfs = []
    for age in age_groups:
        df = pd.read_csv(f"{AGE_DIRECTORY}/{age}.csv")
        df = df[df["year"] >= 1980]
        df = df.drop(["Unnamed: 0"], axis=1)
        freq_dfs.append(df)
    
    # Join
    for age, df in zip(age_groups, freq_dfs):
        df.rename(columns={'freq':'freq_' + age}, inplace=True)
    df_joined = reduce(lambda  left,right: pd.merge(left,right,on=['year', 'word'], how='inner'), freq_dfs)

    # Average across years for each word
    df_joined = df_joined.groupby('word').mean().reset_index()
    df_joined = df_joined.drop(["year"], axis=1)

    # Get offsets
    df_offsets = pd.read_csv(OFFSETS_PATH, sep=',')
    df_offsets = df_offsets.replace(np.nan, 0)
    df_joined = pd.merge(df_joined, df_offsets, how='inner', on=['word'])

    # Level-level Regression
    X = df_joined[['freq_under20', 'freq_20to30', 'freq_30to40', 'freq_40to50', 'freq_over50']]
    y = df_joined['offset']
    X = sm.add_constant(X)
    levlev_model = sm.OLS(y, X).fit()

    # Level-log Regression
    for age in age_groups:
        np.seterr(divide = 'ignore')
        df_joined['log_freq_' + age] = np.where(df_joined['freq_' + age] > 0, np.log2(df_joined['freq_' + age]), 0)
    X = df_joined[['log_freq_under20', 'log_freq_20to30', 'log_freq_30to40', 'log_freq_40to50', 'log_freq_over50']]
    y = df_joined['offset']
    X = sm.add_constant(X)
    levlog_model = sm.OLS(y, X).fit()

    # Log-log Regression
    df_joined['log_offset'] = np.where(df_joined['offset'] > 0, np.log2(df_joined['offset']), 0)
    X = df_joined[['log_freq_under20', 'log_freq_20to30', 'log_freq_30to40', 'log_freq_40to50', 'log_freq_over50']]
    y = df_joined['log_offset']
    X = sm.add_constant(X)
    loglog_model = sm.OLS(y, X).fit()

    # Summaries
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle('Age Regression Results', y=0.92)
    ax1.text(0.01, 0.05, str(levlev_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax1.axis('off')
    ax2.text(0.01, 0.05, str(levlog_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax2.axis('off')
    ax3.text(0.01, 0.05, str(loglog_model.summary()), {'fontsize': 12}, fontproperties = 'monospace') 
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "/age_regression_summaries.png")

    # Regression Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 30))
    fig.suptitle('Age Regression Plots', y=0.92)
    fig = sm.graphics.plot_fit(levlev_model, 0, ax=ax1)
    ax1.set_ylabel("Offset")
    ax1.set_xlabel("Frequencies across age groups")
    ax1.set_title("Level Level Linear Regression")
    fig = sm.graphics.plot_fit(levlog_model, 0, ax=ax2)
    ax2.set_ylabel("Offset")
    ax2.set_xlabel("Frequencies across age groups")
    ax2.set_title("Level Log Linear Regression")
    fig = sm.graphics.plot_fit(loglog_model, 0, ax=ax3)
    ax3.set_ylabel("Log Offset")
    ax3.set_xlabel("Frequencies across age groups")
    ax3.set_title("Log Log Linear Regression")
    plt.savefig(RESULTS_PATH + "/age_regression_plots.png")