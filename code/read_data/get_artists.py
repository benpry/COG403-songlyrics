"""
This file reads data from the musicxmatch dataset in the file "mxm_779k_matches.txt",
determines which are solo artists using US baby names from "NationalNames.csv",
and saves the names of solo artists in the file "artists.csv"
"""

import pandas as pd
import numpy as np
import csv

NAMES_PATH = "../../data/NationalNames.csv"
MXM_PATH = "../../data/mxm_779k_matches.txt"
OUTPUT_PATH = "../../data/processed/artists.csv"

if __name__ == "__main__":

    # Get names from US national names dataset
    data = pd.read_csv(NAMES_PATH, dtype = {'Count': np.int32})
    names = data['Name'].tolist()
    names.remove('The')
    
    # Get all artists in mxm dataset
    all_artists = []
    file = open(MXM_PATH, "r")
    lines = file.readlines()
    for line in lines:
        # Continue if not comment
        if line[0] != "#":
            info = line.rstrip().split("<SEP>")
            # Continue if line has all 6 info fields
            if len(info) == 6:
                # Get mxm artist name from line
                artist_name = info[4]
                # Add to list
                all_artists.append(artist_name)
    file.close()
    # Remove duplicates
    all_artists = list(set(all_artists))

    # Select names of solo artists 
    all_artists_first_names = [name.split(" ")[0] for name in all_artists]
    solo_artists_first_names = list(set(all_artists_first_names).intersection(names))

    solo_artists = []
    for artist in all_artists:
        split = artist.split(" ")
        # If artist name has 2 or fewer words and the first name is a solo name, add to solo artists
        if len(split) <= 2 and split[0] in solo_artists_first_names:
            solo_artists.append(artist)

    # Write to file
    with open(OUTPUT_PATH,"w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(solo_artists)




    