# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display  # noqa
from matplotlib.dates import DayLocator
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

ROOT_PATH = Path("..")
DATA_PATH = ROOT_PATH / "csse_covid_19_data" / "csse_covid_19_time_series"

STATE_COL = "Province/State"
COUNTRY_COL = "Country/Region"
LOCATION_NAME_COL = "Location"
DATE_COL = "Date"
CASES_COL = "Cases"
CASE_TYPE_COL = "Case Type"


def plot(df, style=None):
    style = style or "default"
    with plt.style.context(style):
        # Order locations by decreasing current confirmed case count
        # This is used to keep plot legend in sync with the order of lines on the graph
        # so the location with the most cases is first in the legend and the least is
        # the last
        hue_order = (
            df.groupby(LOCATION_NAME_COL)
            .apply(
                lambda g: pd.Series(
                    {
                        LOCATION_NAME_COL: g.name,
                        CASES_COL: g.loc[
                            g[CASE_TYPE_COL] == "Confirmed", CASES_COL
                        ].iloc[-1],
                    }
                )
            )
            .sort_values(CASES_COL, ascending=False)[LOCATION_NAME_COL]
        )

        g = sns.lineplot(
            x=DATE_COL,
            y=CASES_COL,
            data=df,
            hue=LOCATION_NAME_COL,
            hue_order=hue_order,
            # palette="husl",
            style=CASE_TYPE_COL,
            style_order=["Confirmed", "Recovered", "Deaths"],
        )

        # Y axis setup
        plt.ylim(ymin=1)
        plt.yscale("log", basey=2, nonposy="mask")

        # Configure ticks
        ax = plt.gca()
        ax: plt.Axes
        ax.xaxis.set_minor_locator(DayLocator())
        ax.yaxis.set_major_locator(LogLocator(base=2, numticks=1000))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(
            # 6-2 = 4 minor ticks between each pair of major ticks
            LogLocator(base=2, subs=np.linspace(0.5, 1, 6)[1:-1], numticks=1000)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())

        # Configure design
        plt.setp(g.lines, linewidth=3)
        plt.xticks(rotation=90)
        plt.grid(b=True, which="both", axis="y")
        plt.grid(b=True, which="both", axis="x")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.gcf().set_size_inches((8, 12))
        # plt.gcf().patch.set_facecolor("white")


def get_df(filepath: Path, *, case_type: str):
    df = pd.read_csv(filepath)
    df: pd.DataFrame
    df = df.melt(
        id_vars=[STATE_COL, COUNTRY_COL, "Lat", "Long"],
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

    # Aggregate cities in China
    china = (
        df[df[COUNTRY_COL] == "China"]
        .groupby([DATE_COL, CASE_TYPE_COL], as_index=False)
        .agg({"Lat": "first", "Long": "first", CASES_COL: "sum"})
    )
    china[COUNTRY_COL] = "China"
    df = pd.concat([df, china], axis=0)

    df[COUNTRY_COL] = df[COUNTRY_COL].replace("Korea, South", "South Korea")

    # Use state as location name for states, else use country name
    df[LOCATION_NAME_COL] = df[STATE_COL].fillna(df[COUNTRY_COL])

    # Hereafter df is sorted by date, which is helpful as it allows using .iloc[-1]
    # to get current (or most recent known) situation
    df = df.sort_values([LOCATION_NAME_COL, DATE_COL])
    return df


df = join_dfs()
INCLUDED_COUNTRIES = [
    # "China",
    "Italy",
    "Iran",
    # "United Kingdom",
    "South Korea",
    "US",
]


df = df[
    df[COUNTRY_COL].isin(INCLUDED_COUNTRIES)
    & ~((df[COUNTRY_COL] == "China") & (df[STATE_COL].notna()))
    & (~df[STATE_COL].str.contains(",").fillna(False))
    & (df[DATE_COL] >= pd.to_datetime("2020-02-20"))
]


df = df.groupby(LOCATION_NAME_COL).filter(
    # Exclude US states (unless they meet certain criteria, see below) and
    # UK's discontiguous regions (e.g., Isle of Man, Gibraltar)
    lambda g: g[COUNTRY_COL].iloc[0] not in ["US", "United Kingdom"]
    # Keep UK the country
    or (g.name == "United Kingdom")
    or (
        # Keep top n US states by current number of confirmed cases
        g.loc[g[CASE_TYPE_COL] == "Confirmed", CASES_COL].iloc[-1]
        >= df[df[COUNTRY_COL] == "US"]
        .groupby(LOCATION_NAME_COL)
        .apply(lambda h: h.loc[h[CASE_TYPE_COL] == "Confirmed", CASES_COL].iloc[-1])
        .nlargest(7)
        .iloc[-1]
    )
)

plot(df)
plt.show()
