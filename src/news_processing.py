import polars as pl
import preprocessing
import nltk
from nltk.corpus import stopwords
import argparse


FILE = "data/filtered/04_no_null_articles.parquet"
CLEANED_FILENAME = "data/filtered/05_cleaned_text.parquet"
STEMMED_FILENAME = "data/filtered/06_stemmed_text.parquet"


def main():
    parser = argparse.ArgumentParser(description="Clean and stem news data")
    parser.add_argument("-d", "--dummy_data", action="store_true",
        help="Whether to use dummy data (to test the pipeline) or not")
    args = parser.parse_args()

    lf = pl.scan_parquet(FILE)
    if args.dummy_data:
        lf = lf.head()

    stopword_set = set(stopwords.words("english"))
    stemmer = nltk.stem.SnowballStemmer("english")

    lf = lf.with_columns(
        pl.col("article")
        .map_elements(lambda s: preprocessing.preprocess(s, stopword_set), return_dtype=pl.String)
        .alias("clean_text")
    )

    lf = lf.select(pl.exclude("article")).rename({"clean_text": "text"})
    print("Sinking cleaned parquet file...")
    lf.sink_parquet(CLEANED_FILENAME)
    print(f"Saved preprocessed data to '{CLEANED_FILENAME}'")

    lf = pl.scan_parquet(CLEANED_FILENAME)
    lf = lf.with_columns(
        pl.col("text")
        .map_elements(lambda s: preprocessing.stem(s, stemmer), return_dtype=pl.String)
        .alias("stemmed_text")
    )
    lf = lf.select(pl.exclude("text")).rename({"stemmed_text": "text"})
    print("Sinking stemmed parquet file...")
    lf.sink_parquet(STEMMED_FILENAME)
    print(f"Saved preprocessed data to '{STEMMED_FILENAME}'")


if __name__ == "__main__":
    main()

