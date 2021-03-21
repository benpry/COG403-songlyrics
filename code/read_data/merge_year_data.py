import pandas as pd

DATA_DIR = "../../data/processed"

if __name__ == "__main__":

    # read the date files
    date_dfs = []
    for i in range(4):
        filepath = f"{DATA_DIR}/dates_{i}.csv"
        date_dfs.append(pd.read_csv(filepath))
    df_dates = pd.concat(date_dfs)

    # get the artist and title metadata
    df_lastfm = pd.read_csv(f"{DATA_DIR}/lastfm_metadata.csv")

    df_all_metadata = df_dates.merge(df_lastfm, on="id")
    # replace precise dates with years
    df_all_metadata["date"] = df_all_metadata["date"].apply(lambda x: x.split("-")[0] if type(x) == str else x)
    df_all_metadata.rename(columns={"date": "year"}, inplace=True)

    # save the combined metadata to csv
    df_all_metadata.to_csv(f"{DATA_DIR}/all_metadata.csv", index=False)
