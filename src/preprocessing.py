import polars as pl
import re
import nltk
from nltk.corpus import stopwords
from dotenv import dotenv_values
import pathlib


def main():
    nltk.download("punkt_tab")

    config: dict = dotenv_values(".env") 
    data_dir = pathlib.Path(config["DATA_DIR"])
    df = pl.read_csv(data_dir / "communications.csv")

    stop_words = set(stopwords.words("english"))
    df = df.with_columns(
            pl.col("Text")
            .map_elements(lambda s: preprocess(s, stop_words), return_dtype=pl.String)
            .alias("clean_text")
            )

    stemmer = nltk.stem.SnowballStemmer("english")
    df = df.with_columns(
            pl.col("clean_text")
            .map_elements(lambda s: stem(s, stemmer), return_dtype=pl.String)
            .alias("stemmed_text")
            )

    df.write_csv(data_dir / "communications_preprocessed.csv")
    print("Saved preprocessed data to 'communications_preprocessed.csv'")


def preprocess(text: str, stop_words: set) -> str:
    """
    Performs basic pre-processing.
    Lowercases text, removes non-words and stopwords.

    Parameters
    ----------
    text : str
        String to be preprocessed. Expected to be raw text from `communications.csv`.
    stop_words : set
        Set of English stopwords from nltk.

    Returns
    -------
    str
        Preprocessed string.
    """
    text = re.sub(r'[^a-zA-Z_]', ' ', text) # removes non-words
    words = text.lower().split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)


def stem(text: str, stemmer) -> str:
    """
    Stems the string.

    Parameters
    ----------
    text : str
        String to be stemmed. Expected to be preprocessed text from `communications_preprocessed.csv`.

    Returns
    -------
    str
        String containing stemmed words.
    """
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(t) for t in tokens])


if __name__ == "__main__":
    main()


