import os
import argparse
from tqdm import tqdm
from datetime import datetime
from dotenv import dotenv_values

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Local imports
import frex


CONFIG = dotenv_values(".env")
FILE = f"{CONFIG["DATA_DIR"]}/communications_preprocessed.csv"
WORKERS = 4
MIN_TOPICS = 5
MAX_TOPICS = 30
MODEL_CHOICES = ["lda"]


def main():
    parser = argparse.ArgumentParser(description="Train topic models")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"Number of workers to train algorithms (default: {WORKERS})")
    parser.add_argument("--min_topics", type=int, default=MIN_TOPICS, help=f"Number of topics for smallest model (default: {MIN_TOPICS})")
    parser.add_argument("--max_topics", type=int, default=MAX_TOPICS, help=f"Number of topics for largest number (default: {MAX_TOPICS})")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        choices=MODEL_CHOICES,
        help=f"List of models to run (choices: {MODEL_CHOICES})"
    )
    args = parser.parse_args()

    if args.min_topics > args.max_topics:
        parser.error("min_topics cannot be greater than max_topics")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], timestamp)
    figures_dir = os.path.join(CONFIG["FIGURE_DIR"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = pl.read_csv(FILE)

    texts = [s.split() for s in df["stemmed_text"]]
    gensim_dict = Dictionary(documents=texts)
    corpus = [gensim_dict.doc2bow(t) for t in texts]

    metrics_dict = {"n_topics": [], "exclusivity": [], "coherence": []}

    topic_range = range(args.min_topics, args.max_topics + 1)
    for n in tqdm(topic_range, desc="Training LDA models", unit="model"):
        lda = LdaMulticore(corpus, num_topics=n, workers=args.workers, id2word=gensim_dict)
        topics_df = get_topics_df(lda)

        topics = lda.get_topics()
        frex_metric = frex.exclusivity(topics)
        exclusivity = np.mean(frex_metric) # The original R `plotModels` function computes the average

        cm = CoherenceModel(model=lda, corpus=corpus, texts=texts)
        coherence = cm.get_coherence()

        metrics_dict["n_topics"].append(n)
        metrics_dict["exclusivity"].append(exclusivity)
        metrics_dict["coherence"].append(coherence)

        filename = f"{output_dir}/lda_{n:02}_topics.csv"
        topics_df.write_csv(filename)

    print("All models have been run")
    metrics_df = pl.DataFrame(metrics_dict)
    metrics_filename = f"{output_dir}/lda_metrics.csv"
    metrics_df.write_csv(metrics_filename)
    print(f"Metrics saved to {metrics_filename}")

    plt.figure(figsize=(8, 6))
    plt.scatter(metrics_df["coherence"], metrics_df["exclusivity"])

    # Add text labels for each point (topic number)
    for coherence, exclusivity, n_topics in zip(
        metrics_df["coherence"], metrics_df["exclusivity"], metrics_df["n_topics"]
    ):
        plt.text(coherence, exclusivity, str(n_topics),
             fontsize=9, ha='right', va='bottom')

    plt.xlabel("Coherence")
    plt.ylabel("Exclusivity")
    plt.title("LDA Topics: Coherence vs. Exclusivity")
    plt.grid(True)

    plt.savefig(f"{figures_dir}/lda.png", bbox_inches='tight', dpi=300)
    plt.close()


def get_topics_df(lda) -> pl.DataFrame:
    topics = lda.show_topics(num_topics=-1, formatted=False)
    d = {"word": [], "topic": [], "prob": []}
    for topic_num, word_probs in topics:
        for word, prob in word_probs:
            d["word"].append(word)
            d["topic"].append(topic_num + 1)
            d["prob"].append(prob)
    return pl.DataFrame(d)


if __name__ == "__main__":
    main()

