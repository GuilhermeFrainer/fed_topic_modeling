import polars as pl
import numpy as np
from datetime import date
from typing import NamedTuple


FILE = "data/raw/presidents_and_chairmen.csv"


class Role(NamedTuple):
    name: str
    party: str | None
    start: date
    end: date


def main():
    presidents = [
        Role("George W. Bush", "Republican", date(2001, 1, 20), date(2009, 1, 20)),
        Role("Barack Obama", "Democrat", date(2009, 1, 20), date(2017, 1, 20)),
        Role("Donald Trump", "Republican", date(2017, 1, 20), date(2021, 1, 20)),
        Role("Joe Biden", "Democrat", date(2021, 1, 20), date(2025, 1, 20)),
        Role("Donald Trump", "Republican", date(2025, 1, 20), date(2029, 1, 20))
    ]
    
    chairmen = [
        Role("Alan Greenspan", None, date(1987, 8, 11), date(2006, 2, 1)),
        Role("Ben Bernanke", None, date(2006, 2, 1), date(2014, 2, 1)),
        Role("Janey Yellen", None, date(2014, 2, 1), date(2018, 2, 5)),
        Role("Jerome Powell", None, date(2018,2, 5), date(2029, 1, 1))
    ]
    
    date_range = pl.date_range(
        start=date(2006, 1, 1),
        end=date(2025, 12, 31),
        interval="1d",
        eager=True
    ).rename("date")

    pres_df = get_president_df(date_range, presidents)
    fed_series = get_chairman_series(date_range, chairmen)
    df = pl.DataFrame({
        "date": date_range,
        "president": pres_df["president"],
        "party": pres_df["party"],
        "fed_chair": fed_series
    })
    df.write_csv(FILE)
    print("Data saved to", FILE)


def get_president_df(dates: pl.Series, roles: list[Role]) -> pl.DataFrame:
    names = []
    parties = []
    for d in dates:
        for r in roles:
            if r.start <= d <= r.end:
                names.append(r.name)
                parties.append(r.party)
                break
    return pl.DataFrame({
        "president": names,
        "party": parties
    })


def get_chairman_series(dates: pl.Series, roles: list[Role]) -> pl.Series:
    names = []
    for d in dates:
        for r in roles:
            if r.start <= d <= r.end:
                names.append(r.name)
                break
    series = pl.Series(names)
    series.rename("fed_chairman")
    return series


if __name__ == "__main__":
    main()

