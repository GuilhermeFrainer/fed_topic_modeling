import polars as pl
import pathlib
import sys


NEWS_PATH = r"D:\Projeto NLP\output\stm_news_2025-06-14_14-11-59_topics_13"
FED_PATH = r"D:\Projeto NLP\output\stm_fed_14_2025-06-13_10-32-17"

INTERMEDIARY_NEWS = "news_extended.intermediary.parquet"
INTERMEDIARY_FED = "fed_extended.intermediary.parquet"

FINAL_NEWS = "news_padded.parquet"
FINAL_FED = "fed_padded.parquet"

LARGE_NEGATIVE = -10000


def main():
    news_path = pathlib.Path(NEWS_PATH)
    fed_path = pathlib.Path(FED_PATH)

    try:
        news_df = pl.read_parquet(news_path / INTERMEDIARY_NEWS)
    except FileNotFoundError:
        news_df = pl.read_parquet(news_path / "topic_dist.parquet")

    try:
        fed_df = pl.read_parquet(fed_path / INTERMEDIARY_FED)
    except FileNotFoundError:
        fed_df = pl.read_parquet(fed_path / "topic_dist.parquet")

    try:
        padded_fed_df = pad_dataframe(fed_df, news_df, fed_path / INTERMEDIARY_FED, large_negative_number=LARGE_NEGATIVE)
    except KeyboardInterrupt:
        sys.exit(f"Keyboard interrupt. Fed file saved to {str(fed_path / INTERMEDIARY_FED)}")

    try:
        padded_news_df = pad_dataframe(news_df, fed_df, news_path / INTERMEDIARY_NEWS, large_negative_number=LARGE_NEGATIVE)
    except KeyboardInterrupt:
        padded_fed_df.write_parquet(fed_path / INTERMEDIARY_FED)
        sys.exit(f"Keyboard interrupt. News file saved to {str(news_path / INTERMEDIARY_NEWS)}")

    padded_fed_df.write_parquet(fed_path / FINAL_FED)
    padded_news_df.write_parquet(news_path / FINAL_NEWS)


def pad_dataframe(df1: pl.DataFrame, df2: pl.DataFrame, out_file_path: pathlib.Path, large_negative_number: int = -10000) -> pl.DataFrame:
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    df1_only_cols = df1_cols - df2_cols

    try:
        for col in df1_only_cols:
            df1 = df1.with_columns(pl.lit(-1000).alias(col))
    except KeyboardInterrupt:
        df1 = df1.select(sorted(df1.columns))
        df1.write_parquet(out_file_path)
        raise KeyboardInterrupt
    return df1

if __name__ == "__main__":
    main()

