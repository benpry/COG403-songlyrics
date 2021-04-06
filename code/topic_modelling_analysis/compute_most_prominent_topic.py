"""
This file separates songs by their most prominent topic, then computes how much ahead or or behind songs are when they
have each topic as their most prominent
"""
import sys
sys.path.append("../read_data")
from compute_aggregate_song_frequencies import compute_frequency_dataframe
from collections import defaultdict
import numpy as np
from scipy.stats import norm
import pickle

N_TOPICS = 30
SONG_FILEPATH = "../../data/processed/songs_with_topics_english.p"
OUTPUT_FILEPATH = "../../data/processed/songs_with_top_topics.p"
FREQUENCY_FOLDER = "../../data/processed/freqs_by_topic"

if __name__ == "__main__":

    with open(SONG_FILEPATH, "rb") as fp:
        songs = pickle.load(fp)

    topic_probs = defaultdict(list)
    for song in songs:
        for i in range(N_TOPICS):
            if i in song["topics"]:
                topic_probs[i].append(song["topics"][i])
            else:
                topic_probs[i].append(0)

    topic_distributions = {}
    for i in range(N_TOPICS):
        topic_distributions[i] = (np.mean(topic_probs[i]), np.std(topic_probs[i]))

    for song in songs:
        top_topic = -1
        top_prob = -9e10
        for topic_idx in song["topics"]:
            stdev, mean = topic_distributions[topic_idx]
            normalized_prob = (song["topics"][topic_idx] - mean) / stdev
            if normalized_prob > top_prob:
                top_topic = topic_idx
                top_mixture = normalized_prob
        song["top_topic"] = top_topic

    with open(OUTPUT_FILEPATH, "wb") as fp:
        pickle.dump(songs, fp)

    for topic_idx in range(N_TOPICS):
        topic_songs = [song for song in songs if song["top_topic"] == topic_idx]

        df_topic = compute_frequency_dataframe(topic_songs)
        df_topic.to_csv(f"{FREQUENCY_FOLDER}/topic_{topic_idx}.csv")
