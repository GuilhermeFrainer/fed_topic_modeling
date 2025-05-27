import os
import argparse
import matplotlib.figure
from tqdm import tqdm
from datetime import datetime
from dotenv import dotenv_values
import pathlib

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib

from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import nltk

# Local imports
import frex
import preprocessing


CONFIG = dotenv_values(".env")
DATASETS = ["fed", "newsgroups"]
# Default CLI options
WORKERS = 4
MIN_TOPICS = 5
MAX_TOPICS = 30


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = "lda" + "_" + args.dataset + "_" + timestamp
    data_dir = pathlib.Path(os.path.join(CONFIG["DATA_DIR"], "processed"))
    output_dir = os.path.join(CONFIG["OUTPUT_DIR"], dir_name)
    figures_dir = os.path.join(CONFIG["FIGURE_DIR"], dir_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    texts = load_dataset(args.dataset, data_dir)
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

    fig = plot_coherence_and_exclusivity(metrics_df)
    fig.savefig(f"{figures_dir}/lda.png", bbox_inches='tight', dpi=300)
    plt.close(fig)


def parse_args():
    """
    Instantiates parser and sets up arguments. Returns parsed args.
    """
    parser = argparse.ArgumentParser(description="Train topic models")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"Number of workers to train algorithms (default: {WORKERS})")
    parser.add_argument("--min_topics", type=int, default=MIN_TOPICS, help=f"Number of topics for smallest model (default: {MIN_TOPICS})")
    parser.add_argument("--max_topics", type=int, default=MAX_TOPICS, help=f"Number of topics for largest number (default: {MAX_TOPICS})")
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="fed",
        help=f"Dataset to be used to train the models (choices: {DATASETS})")
    args = parser.parse_args()

    if args.min_topics > args.max_topics:
        parser.error("min_topics cannot be greater than max_topics")
    return args


def load_dataset(dataset: str, data_dir: pathlib.Path) -> list[list[str]]:
    """
    Loads chosen dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    data_dir : pathlib.Path
        Path to the data.
    
    Returns
    -------
    list[list[str]]
        List of documents. Each document is stripped, so it's a list of strings.
    """
    if dataset == "fed":
        filename = data_dir / "communications.csv"
        df = pl.read_csv(filename)
        texts = [s.split() for s in df["stemmed_text"]]
    elif dataset == "newsgroups":
        data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
        stop_words = set(stopwords.words("english"))
        stemmer = nltk.stem.SnowballStemmer("english")

        preprocessed_texts = [preprocessing.preprocess(t, stop_words=stop_words) for t in data["data"]]
        stemmed_texts = [preprocessing.stem(t, stemmer=stemmer) for t in preprocessed_texts]
        texts = [t.split() for t in stemmed_texts]
    return texts


def get_topics_df(lda) -> pl.DataFrame:
    topics = lda.show_topics(num_topics=-1, formatted=False)
    d = {"word": [], "topic": [], "prob": []}
    for topic_num, word_probs in topics:
        for word, prob in word_probs:
            d["word"].append(word)
            d["topic"].append(topic_num + 1)
            d["prob"].append(prob)
    return pl.DataFrame(d)


def plot_coherence_and_exclusivity(df: pl.DataFrame) -> matplotlib.figure.Figure:
    """
    Plots coherence and exclusivity chart.

    Paramters
    ---------
    df : pl.DataFrame
        DataFrame containing coherence and exclusivity metrics for each topic model.

    Returns
    -------
    matplotlib.figure.Figure
        Plot of coherence and exclusivity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["coherence"], df["exclusivity"])

    # Add text labels for each point (topic number)
    for coherence, exclusivity, n_topics in zip(
        df["coherence"], df["exclusivity"], df["n_topics"]
    ):
        ax.text(coherence, exclusivity, str(n_topics),
                fontsize=9, ha='right', va='bottom')

    ax.set_xlabel("Coherence")
    ax.set_ylabel("Exclusivity")
    ax.set_title("Coherence vs. Exclusivity")
    ax.grid(True)
    return fig


if __name__ == "__main__":
    main()

