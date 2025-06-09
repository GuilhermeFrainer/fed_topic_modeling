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
    raw_data = data_dir / "raw"
    processed_data = data_dir / "processed"

    communications_raw_path = processed_data / "communications_raw.parquet"
    communications_clean_path = processed_data / "communications_clean.parquet"
    communications_stemmed_path = processed_data / "communications_stemmed.parquet"

    # Converting to parquet
    df = pl.read_csv(raw_data / "communications.csv")
    df = df.rename({"Text": "text"})
    df.write_parquet(communications_raw_path)

    # Removing numbers, stopwords
    lf = pl.scan_parquet(processed_data / "communications_raw.parquet")
    stop_words = set(stopwords.words("english"))
    lf = lf.with_columns(
            pl.col("text")
            .map_elements(lambda s: preprocess(s, stop_words), return_dtype=pl.String)
            .alias("clean_text")
            )

    lf = lf.select(pl.exclude("text")).rename({"clean_text": "text"})
    print("Sinking cleaned parquet file...")
    lf.sink_parquet(communications_clean_path)
    print(f"Saved preprocessed data to '{str(communications_clean_path)}'")

    # Stemming
    lf = pl.scan_parquet(communications_clean_path)
    stemmer = nltk.stem.SnowballStemmer("english")
    lf = lf.with_columns(
            pl.col("text")
            .map_elements(lambda s: stem(s, stemmer), return_dtype=pl.String)
            .alias("stemmed_text")
            )

    lf = lf.select(pl.exclude("text")).rename({"stemmed_text": "text"})
    print("Sinking stemmed parquet...")
    lf.sink_parquet(communications_stemmed_path)
    print(f"Saved preprocessed data to '{str(communications_stemmed_path)}'")


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


