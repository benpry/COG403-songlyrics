"""
Print the top few words for each topic
"""
from gensim.models.ldamodel import LdaModel

MODEL_PATH = "../../data/lda/lda_model.bin"

if __name__ == "__main__":

    lda = LdaModel.load(MODEL_PATH)

    for t in range(30):
        print(f"topic {t}")
        for term in lda.get_topic_terms(t):
            print(term)

        print("\n")
