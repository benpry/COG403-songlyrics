"""
This file tests whether each song is in english or not by seeing whether it contains some of the most common english
words. It filters out all of the non-english words.
"""
import pickle
from copy import deepcopy
import sys

ENGLISH_WORD_PATH = "../../data/top_english_words.txt"

if __name__ == "__main__":

    filename = sys.argv[1]

    # load the songs
    with open(filename, "rb") as fp:
        songs = pickle.load(fp)

    with open(ENGLISH_WORD_PATH, "r") as fp:
        english_words = set([x.replace("\n", "") for x in fp.readlines()])

    new_songs = []
    for song in songs:
        words = set(song["word_freqs"].keys())

        english_freq = 0
        non_english_freq = 0
        for word in song["word_freqs"]:
            if word in english_words:
                english_freq += song["word_freqs"][word]
            else:
                non_english_freq += song["word_freqs"][word]

        english_percent = english_freq / (english_freq + non_english_freq)

        if english_percent < 0.8:
            continue

        new_songs.append(song)

    output_filename = filename[:-2] + "_english.p"
    with open(output_filename, "wb") as fp:
        pickle.dump(new_songs, fp)
