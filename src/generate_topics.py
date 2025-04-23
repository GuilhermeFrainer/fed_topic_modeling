import polars as pl
import numpy as np
from dotenv import dotenv_values
import pathlib
from gensim.models.ldamulticore import LdaMulticore
import gensim.models.basemodel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import warnings


CONFIG = dotenv_values(".env")
FILE = f"{CONFIG["DATA_DIR"]}/communications_preprocessed.csv"
WORKERS = 4
MIN_TOPICS = 5
MAX_TOPICS = 30


def main():
    df = pl.read_csv(FILE)

    texts = [s.split() for s in df["stemmed_text"]]
    gensim_dict = Dictionary(documents=texts)
    corpus = [gensim_dict.doc2bow(t) for t in texts]

    metrics_dict = {"n_topics": [], "exclusivity": [], "coherence": []}

    warnings.warn("Exclusivity metrics are calculated improperly for now")
    for n in range(MIN_TOPICS, MAX_TOPICS + 1):
        print(f"Training model with {n} topics")
        lda = LdaMulticore(corpus, num_topics=n, workers=WORKERS, id2word=gensim_dict)
        topics = get_topics_df(lda)

        exclusivity = get_exclusivity(lda).sum()

        print("Computing coherence")
        cm = CoherenceModel(model=lda, corpus=corpus, texts=texts)
        coherence = cm.get_coherence()

        metrics_dict["n_topics"].append(n)
        metrics_dict["exclusivity"].append(exclusivity)
        metrics_dict["coherence"].append(coherence)

        filename = f"{CONFIG["OUTPUT_DIR"]}/lda_{n:02}_topics.csv"
        topics.write_csv(filename)
        print(f"Results saved to {filename}")

    print("All models have been run")
    metrics_df = pl.DataFrame(metrics_dict)
    metrics_filename = f"{CONFIG["OUTPUT_DIR"]}/lda_metrics.csv"
    metrics_df.write_csv(metrics_filename)
    print(f"Metrics saved to {metrics_filename}")


def get_topics_df(lda) -> pl.DataFrame:
    topics = lda.show_topics(formatted=False)
    d = {"word": [], "topic": [], "prob": []}
    for topic_num, word_probs in topics:
        for word, prob in word_probs:
            d["word"].append(word)
            d["topic"].append(topic_num + 1)
            d["prob"].append(prob)
    return pl.DataFrame(d)


def get_exclusivity(model: gensim.models.basemodel.BaseTopicModel) -> np.ndarray: 
    topic_word_probs: np.ndarray = model.get_topics()
    word_totals = topic_word_probs.sum(axis=0)
    word_totals[word_totals == 0] = 1e-10 # Avoids division by zero
    return topic_word_probs / word_totals


if __name__ == "__main__":
    main()

