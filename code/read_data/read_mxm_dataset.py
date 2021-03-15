"""
This file reads the MusixMatch dataset and saves it into a convenient file for future analysis
"""
import pickle
import pandas as pd

MXM_PATH = "../../data/mxm_dataset_train.txt"
OUTPUT_PATH = "../../data/processed/songs.p"
ALL_WORDS_PATH = "../../data/processed/allwords.csv"

if __name__ == "__main__":

    all_songs = []

    fp = open(MXM_PATH, "r", encoding="utf-8")

    lines = fp.readlines()
    all_words = lines[17][1:-1].split(",")

    for line in lines[18:]:
        line = line.split(",")

        line_id = line[0]
        line_mxmid = line[1]
        print(line_mxmid)

        word_freqs = {}
        for word_freq in line[2:]:
            word_idx, freq = word_freq.split(":")
            word_idx = int(word_idx) - 1
            freq = int(freq)
            word_freqs[all_words[word_idx]] = freq

        all_songs.append({"id": line_id, "mmx_id": line_mxmid, "word_freqs": word_freqs})

    with open(OUTPUT_PATH, "wb") as fp:
        pickle.dump(all_songs, fp)

    pd.DataFrame({"word": all_words}).to_csv(ALL_WORDS_PATH, sep=",", encoding="utf-8")
