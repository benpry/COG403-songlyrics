"""
This does the regression analysis by age
"""

import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.api as sm
import matplotlib.pyplot as plt

AGE_DIRECTORY = "../../data/processed/freqs_by_age"
AGGREGATE_RESULTS_PATH = "../../data/aggregate_results.csv"
RESULTS_PATH = "../../figures"

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

    # Add offsets and granger lags
    df_offsets_lags = pd.read_csv(AGGREGATE_RESULTS_PATH, sep=',')
    df_offsets_lags = df_offsets_lags.replace(np.nan, 0)
    df_joined = pd.merge(df_joined, df_offsets_lags, how='inner', on=['word'])

    # Level-level Regression
    X = df_joined[['freq_' + age for age in age_groups]]
    X = sm.add_constant(X)
    y1 = df_joined['offset']
    y2 = df_joined['granger_lag']
    levlev_model_offset = sm.OLS(y1, X).fit()
    levlev_model_granger = sm.OLS(y2, X).fit()

    # Level-log Regression
    for age in age_groups:
        np.seterr(divide = 'ignore')
        df_joined['log_freq_' + age] = np.where(df_joined['freq_' + age] > 0, np.log2(df_joined['freq_' + age]), 0)
    X = df_joined[['log_freq_' + age for age in age_groups]]
    X = sm.add_constant(X)
    y1 = df_joined['offset']
    y2 = df_joined['granger_lag']
    levlog_model_offset = sm.OLS(y1, X).fit()
    levlog_model_granger = sm.OLS(y2, X).fit()

    # Log-log Regression
    df_joined['log_offset'] = np.where(df_joined['offset'] > 0, np.log2(df_joined['offset']), 0)
    df_joined['log_granger_lag'] = np.where(df_joined['granger_lag'] > 0, np.log2(df_joined['granger_lag']), 0)
    X = df_joined[['log_freq_' + age for age in age_groups]]
    X = sm.add_constant(X)
    y1 = df_joined['log_offset']
    y2 = df_joined['log_granger_lag']
    loglog_model_offset = sm.OLS(y1, X).fit()
    loglog_model_granger = sm.OLS(y2, X).fit()

    # Offset Summaries
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle('Offset on Age Regression Results', y=0.92)
    ax1.text(0.01, 0.05, str(levlev_model_offset.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax1.axis('off')
    ax2.text(0.01, 0.05, str(levlog_model_offset.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax2.axis('off')
    ax3.text(0.01, 0.05, str(loglog_model_offset.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "/age_offset_regression_summaries.png")

    # Granger Lag Summaries
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 24))
    fig.suptitle('Granger Lag on Age Regression Results', y=0.92)
    ax1.text(0.01, 0.05, str(levlev_model_granger.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax1.axis('off')
    ax2.text(0.01, 0.05, str(levlog_model_granger.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax2.axis('off')
    ax3.text(0.01, 0.05, str(loglog_model_granger.summary()), {'fontsize': 12}, fontproperties = 'monospace')
    ax3.axis('off')
    plt.tight_layout()
    plt.savefig(RESULTS_PATH + "/age_granger_regression_summaries.png")
