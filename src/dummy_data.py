from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import polars as pl
import nltk
import preprocessing


def main():
    stop_words = set(stopwords.words("english"))
    stemmer = nltk.stem.SnowballStemmer("english")

    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        categories=["sci.space", "rec.sport.baseball"])

    sample = data["data"][:10]
    preprocessed_text = [preprocessing.preprocess(s, stop_words=stop_words) for s in sample]
    stemmed_text = [preprocessing.stem(s, stemmer=stemmer) for s in preprocessed_text]

    df = pl.DataFrame({"raw_text": sample, "clean_text": preprocessed_text, "stemmed_text": stemmed_text})
    df.write_csv("../data/dummy_data.csv")


if __name__ == "__main__":
    main()

