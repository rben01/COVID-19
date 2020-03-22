# %%
import itertools
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display  # noqa
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from constants import (
    CASE_COUNT_COL,
    CASE_TYPE_COL,
    CONFIRMED,
    COUNTRY_COL,
    DATE_COL,
    DEATHS,
    LATITUDE_COL,
    LOCATION_NAME_COL,
    LONGITUDE_COL,
    RECOVERED,
    ROOT_PATH,
    STATE_COL,
)
from get_worldwide_cases import get_worldwide_case_count

DATA_PATH = ROOT_PATH / "csse_covid_19_data" / "csse_covid_19_time_series"


def plot(df, *, style=None, start_date=None, include_recovered=False):
    worldwide_case_count = get_worldwide_case_count()

    if start_date is not None:
        df = df[df[DATE_COL] >= pd.Timestamp(start_date)]
        worldwide_case_count = worldwide_case_count[
            worldwide_case_count[DATE_COL] >= pd.Timestamp(start_date)
        ]

    if not include_recovered:
        df = df[df[CASE_TYPE_COL] != RECOVERED]
        worldwide_case_count = worldwide_case_count[
            worldwide_case_count[CASE_TYPE_COL] != RECOVERED
        ]

    current_case_counts = (
        df.groupby(LOCATION_NAME_COL)
        .apply(
            lambda g: pd.Series(
                {
                    LOCATION_NAME_COL: g.name,
                    **{
                        case_type: g.loc[g[CASE_TYPE_COL] == case_type, CASE_COUNT_COL]
                        .tail(1)
                        .sum()
                        for case_type in [CONFIRMED, RECOVERED, DEATHS]
                    },
                }
            )
        )
        .sort_values(CONFIRMED, ascending=False)
    )

    hue_order = current_case_counts[LOCATION_NAME_COL]

    case_type_names, dashes = zip(
        *itertools.compress(
            *zip(
                ((CONFIRMED, (1, 0)), True),
                ((RECOVERED, (3, 2, 1, 2)), include_recovered),
                ((DEATHS, (1, 1)), True),
            )
        )
    )

    style = style or "default"
    with plt.style.context(style):
        # Order locations by decreasing current confirmed case count
        # This is used to keep plot legend in sync with the order of lines on the graph
        # so the location with the most cases is first in the legend and the least is
        # the last

        g = sns.lineplot(
            data=df,
            x=DATE_COL,
            y=CASE_COUNT_COL,
            hue=LOCATION_NAME_COL,
            hue_order=hue_order,
            # palette="husl",
            style=CASE_TYPE_COL,
            style_order=case_type_names,
            dashes=dashes,
        )

        # Configure axes and ticks
        ax = plt.gca()
        ax: plt.Axes
        ax.set_ylim(bottom=1)
        ax.set_yscale("log", basey=2, nonposy="mask")
        ax.xaxis.set_minor_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %-d"))
        ax.yaxis.set_major_locator(LogLocator(base=2, numticks=1000))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(
            # 6-2 = 4 minor ticks between each pair of major ticks
            LogLocator(base=2, subs=np.linspace(0.5, 1, 6)[1:-1], numticks=1000)
        )
        ax.yaxis.set_minor_formatter(NullFormatter())

        # Configure design
        plt.gcf().set_size_inches((12, 12))
        for line in g.lines:
            line.set_linewidth(3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(80)
        ax.grid(b=True, which="both", axis="both")
        legend = plt.legend(loc="upper left", framealpha=0.9)

        # Add case counts of the different categories to the legend (next few blocks)
        sep_str = " / "
        left_str = " ("
        right_str = ")"

        # Add number format to legend title (the first text in the legend)
        fmt_str = sep_str.join(case_type_names)
        next(iter(legend.texts)).set_text(f"Location{left_str}{fmt_str}{right_str}")

        # Add case counts to legend labels (first label is title, so skip it)
        case_count_str_cols = [
            current_case_counts[col].map("{:,}".format) for col in case_type_names
        ]
        labels = (
            current_case_counts[LOCATION_NAME_COL]
            + left_str
            + case_count_str_cols[0].str.cat(case_count_str_cols[1:], sep=sep_str)
            + right_str
        )
        for text, label in zip(itertools.islice(legend.texts, 1, None), labels):
            text.set_text(label)

        # Add worldwide case count to right y axis
        # wwcc_ax = ax.twinx()
        # wwcc_ax: plt.Axes
        # wwcc_ax.scatter(
        #     x=worldwide_case_count[DATE_COL],
        #     y=worldwide_case_count[CASE_COUNT_COL],
        #     c="black",
        #     marker="o",
        # )
        # wwcc_ax.set_ylabel("Worldwide Cases")

        # wwcc_ax.set_yscale("log", basey=10, nonposy="mask")
        # wwcc_ax.yaxis.set_major_formatter(ScalarFormatter())


def get_df(filepath: Path, *, case_type: str):
    df = pd.read_csv(filepath)
    df: pd.DataFrame
    df = df.melt(
        id_vars=[STATE_COL, COUNTRY_COL, LATITUDE_COL, LONGITUDE_COL],
        var_name=DATE_COL,
        value_name=CASE_COUNT_COL,
    )
    df[CASE_TYPE_COL] = case_type

    return df


def join_dfs():
    dfs = []
    dfs: List[pd.DataFrame]
    for csv in DATA_PATH.glob("*.csv"):
        case_type = csv.stem.replace("time_series_19-covid-", "")
        df = get_df(csv, case_type=case_type)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Remove cities in US (eg New York, NY)
    df = df[~df[STATE_COL].str.contains(",").fillna(False)]

    # Aggregate cities/province/states in select countries
    agg_country_df = (
        df[df[COUNTRY_COL].isin(["China", "US"])]
        .groupby([DATE_COL, CASE_TYPE_COL, COUNTRY_COL], as_index=False)
        .agg({LATITUDE_COL: "first", LONGITUDE_COL: "first", CASE_COUNT_COL: "sum"})
    )

    df = pd.concat([df, agg_country_df], axis=0)

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
    "Spain",
    "Germany",
    # "United Kingdom",
    "South Korea",
    "US",
]


df = df[
    df[COUNTRY_COL].isin(INCLUDED_COUNTRIES)
    & ~((df[COUNTRY_COL] == "China") & (df[STATE_COL].notna()))
]

df = df.groupby(LOCATION_NAME_COL).filter(
    # Exclude US states (unless they meet certain criteria, see below) and
    # UK's discontiguous regions (e.g., Isle of Man, Gibraltar)
    lambda g: g[COUNTRY_COL].iloc[0] not in ["US", "United Kingdom"]
    # Keep UK the country
    or (g.name == "United Kingdom")
    or (
        # Keep top n US states by current number of confirmed cases
        g.loc[g[CASE_TYPE_COL] == CONFIRMED, CASE_COUNT_COL].iloc[-1]
        >= df[(df[COUNTRY_COL] == "US") & (df[LOCATION_NAME_COL] != "US")]
        .groupby(LOCATION_NAME_COL)
        .apply(lambda h: h.loc[h[CASE_TYPE_COL] == CONFIRMED, CASE_COUNT_COL].iloc[-1])
        .nlargest(4)
        .iloc[-1]
    )
)

plot(df, start_date="2020-02-20", include_recovered=False)
plt.show()
