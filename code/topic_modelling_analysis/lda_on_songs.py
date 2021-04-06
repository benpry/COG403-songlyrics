"""
This file applies Latent Dirichlet Allocation to identify topics in songs
"""
from gensim.models.ldamodel import LdaModel
from copy import deepcopy
from gensim import corpora
from gensim.utils import simple_preprocess
import pickle

OUTPUT_FILE = "../../data/lda/lda_model.bin"
SONGS_INPUT_FILE = "../../data/processed/songs_with_metadata_english.p"
SONGS_OUTPUT_FILE = "../../data/processed/songs_with_topics_english.p"
TOPIC_WORDS_FILE = "../../data/lda_topic_words.txt"
n_topics = 30

if __name__ == "__main__":

    with open(SONGS_INPUT_FILE, "rb") as fp:
        songs = pickle.load(fp)

    docs_tokenized = []
    for song in songs:
        word_dict = song["word_freqs"]
        song_str = ""
        for word in word_dict:
            song_str += (word + " ") * word_dict[word]
        docs_tokenized.append(simple_preprocess(song_str))

    dictionary = corpora.Dictionary()
    bow_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in docs_tokenized]

    lda = LdaModel(bow_corpus, n_topics)

    with open(TOPIC_WORDS_FILE, "w") as fp:
        for topic_id in range(n_topics):
            print(f"topic {topic_id} terms:")
            fp.write(f"\ntopic {topic_id} terms:\n")
            for term in lda.get_topic_terms(topic_id):
                fp.write(f"{dictionary.get(term[0])}: p={term[1]}\n")
                print(f"{dictionary.get(term[0])}: p={term[1]}")

    # save the LDA model
    # lda.save(OUTPUT_FILE)

    # add topic mixtures to each song
    new_songs = []
    for song in songs:
        word_dict = song["word_freqs"]
        song_str = ""
        for word in word_dict:
            song_str += (word + " ") * word_dict[word]
        song_tokenized = simple_preprocess(song_str)
        song_bow = dictionary.doc2bow(song_tokenized)
        topics = lda.get_document_topics(song_bow)

        new_song = deepcopy(song)
        new_song["topics"] = {}
        for topic in topics:
            new_song["topics"][topic[0]] = topic[1]
        new_songs.append(new_song)

    with open(SONGS_OUTPUT_FILE, "wb") as fp:
        pickle.dump(new_songs, fp)
