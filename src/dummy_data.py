from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from dotenv import dotenv_values
import polars as pl
import nltk
import preprocessing
import pathlib


ROWS = 10
FILE = "dummy.csv"


def main():
    config = dotenv_values(".env")
    data_dir = pathlib.Path(config["DATA_DIR"]) / "processed"

    stop_words = set(stopwords.words("english"))
    stemmer = nltk.stem.SnowballStemmer("english")

    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        categories=["sci.space", "rec.sport.baseball"])

    sample = data["data"][:ROWS]
    preprocessed_text = [preprocessing.preprocess(s, stop_words=stop_words) for s in sample]
    stemmed_text = [preprocessing.stem(s, stemmer=stemmer) for s in preprocessed_text]

    df = pl.DataFrame({"raw_text": sample, "clean_text": preprocessed_text, "stemmed_text": stemmed_text})
    filename = data_dir / FILE
    df.write_csv(filename)
    print("Dummy data file saved to " + str(filename))


if __name__ == "__main__":
    main()

