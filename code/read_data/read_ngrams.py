"""
This file reads the ngrams from google ngram viewer.
"""
from getngrams import getNgrams
import pickle
import requests

ALL_WORDS_PATH = "../../data/processed/allwords.p"

if __name__ == "__main__":

    with open(ALL_WORDS_PATH, "rb") as fp:
        all_words = pickle.load(fp)

    corpus, startYear, endYear, smoothing = 'eng_2012', 1800, 2000, 3
    printHelp, caseInsensitive, allData = False, True, False

    for word in all_words:

        url, content, df = getNgrams(word, corpus, startYear, endYear, smoothing, caseInsensitive)

        print(url)
