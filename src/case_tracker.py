# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display  # noqa
from matplotlib.dates import DayLocator
from matplotlib.ticker import ScalarFormatter


ROOT_PATH = Path("..")
DATA_PATH = ROOT_PATH / "csse_covid_19_data" / "csse_covid_19_time_series"

_STATE_COL = "Province/State"
_COUNTRY_COL = "Country/Region"
_SORT_ORDER_COL = "Sort Order"
LOCATION_NAME_COL = "Location"
DATE_COL = "Date"
CASES_COL = "Cases"
CASE_TYPE_COL = "Case Type"


def plot(df, style=None):
    style = style or "seaborn-deep"
    with plt.style.context(style):
        g = sns.lineplot(
            x=DATE_COL,
            y=CASES_COL,
            data=df,
            hue=LOCATION_NAME_COL,
            style=CASE_TYPE_COL,
            style_order=["Confirmed", "Recovered", "Deaths"],
        )
        plt.xticks(rotation=90)
        plt.yscale("log", basey=8, nonposy="mask")
        plt.ylim(ymin=1)
        plt.gca().xaxis.set_minor_locator(DayLocator())
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
        plt.grid(b=True, which="both", axis="y")
        plt.grid(b=True, which="both", axis="x")
        plt.gcf().set_size_inches((15, 20))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.gcf().patch.set_facecolor("white")


def get_df(filepath: Path, *, case_type: str):
    df = pd.read_csv(filepath)
    df: pd.DataFrame
    df = df.melt(
        id_vars=[_STATE_COL, _COUNTRY_COL, "Lat", "Long"],
        var_name=DATE_COL,
        value_name=CASES_COL,
    )
    df[CASE_TYPE_COL] = case_type
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return df


def join_dfs():
    dfs = []
    dfs: List[pd.DataFrame]
    for csv in DATA_PATH.glob("*.csv"):
        case_type = csv.stem.replace("time_series_19-covid-", "")
        df = get_df(csv, case_type=case_type)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    china = (
        df[df[_COUNTRY_COL] == "China"]
        .groupby([DATE_COL, CASE_TYPE_COL], as_index=False)
        .agg({"Lat": "first", "Long": "first", CASES_COL: "sum"})
    )
    china[_COUNTRY_COL] = "China"
    df = pd.concat([df, china], axis=0)

    df[LOCATION_NAME_COL] = df[_STATE_COL].fillna(df[_COUNTRY_COL])
    return df


df = join_dfs()
INCLUDED_COUNTRIES = [
    # "China",
    "Italy",
    # "Iran",
    # "United Kingdom",
    "Korea, South",
    "US",
]


df = df[
    df[_COUNTRY_COL].isin(INCLUDED_COUNTRIES)
    & ~((df[_COUNTRY_COL] == "China") & (df[_STATE_COL].notna()))
    & (~df[_STATE_COL].str.contains(",").fillna(False))
    & (df[DATE_COL] >= pd.to_datetime("2020-02-20"))
]


df = df.groupby(LOCATION_NAME_COL).filter(
    lambda g: g[_COUNTRY_COL].iloc[0] not in ["US", "United Kingdom"]
    or (g.name == "United Kingdom")
    or (
        g.loc[g[CASE_TYPE_COL] == "Confirmed", CASES_COL].max()
        >= df[df[_COUNTRY_COL] == "US"]
        .groupby(LOCATION_NAME_COL)
        .apply(lambda h: h.loc[h[CASE_TYPE_COL] == "Confirmed", CASES_COL].max())
        .nlargest(7)
        .iloc[-1]
    )
)

df = df.merge(
    df.groupby(LOCATION_NAME_COL)
    .apply(
        lambda g: g[g[CASE_TYPE_COL] == "Confirmed"]
        .sort_values(DATE_COL)[CASES_COL]
        .iloc[-1]
    )
    .rename(_SORT_ORDER_COL),
    on=LOCATION_NAME_COL,
    how="left",
)

df = df.sort_values(_SORT_ORDER_COL, ascending=False)

plot(df)
