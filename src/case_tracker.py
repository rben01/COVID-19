# %%
import itertools
from collections import namedtuple
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display  # noqa
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from constants import CaseTypes, Columns, Paths, Strings

DATA_PATH = Paths.ROOT / "csse_covid_19_data" / "csse_covid_19_time_series"

CASE_TYPE_NAMES = "name"
DASH_STYLE = "dash_style"
INCLUDE = "include"
PerCaseTypeConfig = namedtuple(
    "PerCaseTypeConfig", [CASE_TYPE_NAMES, DASH_STYLE, INCLUDE]
)


def _plot_helper(
    df: pd.DataFrame,
    *,
    style=None,
    filter_df_func,
    palette=None,
    config: List[PerCaseTypeConfig],
    plot_size: Tuple[float],
):
    plt.subplots()

    df = filter_df_func(df)

    current_case_counts = (
        df.groupby(Columns.LOCATION_NAME).apply(
            lambda g: pd.Series(
                {
                    Columns.LOCATION_NAME: g.name,
                    # Get last case count of each case type for each location
                    **g.groupby(Columns.CASE_TYPE)[Columns.CASE_COUNT]
                    # .tail(1).sum() is a hack to get the last value if it exists else 0
                    .apply(lambda h: h.tail(1).sum()).to_dict(),
                }
            )
        )
        # Order locations by decreasing current confirmed case count
        # This is used to keep plot legend in sync with the order of lines on the graph
        # so the location with the most current cases is first in the legend and the
        # least is last
        .sort_values(CaseTypes.CONFIRMED, ascending=False)
    )
    current_case_counts[CaseTypes.MORTALITY] = (
        current_case_counts[CaseTypes.DEATHS] / current_case_counts[CaseTypes.CONFIRMED]
    )

    hue_order = current_case_counts[Columns.LOCATION_NAME]

    config_df = pd.DataFrame(config)
    config_df = config_df[config_df[INCLUDE]]
    legend_fields = [*config_df[CASE_TYPE_NAMES], CaseTypes.MORTALITY]

    style = style or "default"
    with plt.style.context(style):
        g = sns.lineplot(
            data=df,
            x=Columns.DATE,
            y=Columns.CASE_COUNT,
            hue=Columns.LOCATION_NAME,
            hue_order=hue_order,
            style=Columns.CASE_TYPE,
            style_order=config_df[CASE_TYPE_NAMES].tolist(),
            dashes=config_df[DASH_STYLE].tolist(),
            palette=None,
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
        plt.gcf().set_size_inches(plot_size)
        for line in g.lines:
            line.set_linewidth(3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(80)
        ax.grid(b=True, which="both", axis="both")
        legend = plt.legend(loc="best", framealpha=0.9)
        plt.gcf().set_facecolor("white")

        # Add case counts of the different categories to the legend (next few blocks)
        sep_str = " / "
        left_str = " ("
        right_str = ")"

        # Add number format to legend title (the first text in the legend)
        fmt_str = sep_str.join(legend_fields)
        next(iter(legend.texts)).set_text(f"Location{left_str}{fmt_str}{right_str}")

        # Add case counts to legend labels (first label is title, so skip it)
        case_count_str_cols = [
            current_case_counts[col].map(r"{:,}".format)
            for col in config_df[CASE_TYPE_NAMES]
        ]
        case_count_str_cols.append(
            current_case_counts[CaseTypes.MORTALITY].map(r"{0:.2%}".format)
        )
        labels = (
            current_case_counts[Columns.LOCATION_NAME]
            + left_str
            + case_count_str_cols[0].str.cat(case_count_str_cols[1:], sep=sep_str)
            + right_str
        )
        for text, label in zip(itertools.islice(legend.texts, 1, None), labels):
            text.set_text(label)


def plot_world_and_china(df: pd.DataFrame, *, style=None, start_date=None):
    def filter_func(df):
        return df[
            df[Columns.LOCATION_NAME].isin(
                [Strings.WORLD, Strings.WORLD_MINUS_CHINA, "China"]
            )
            & (df[Columns.CASE_TYPE] != CaseTypes.RECOVERED)
            & (df[Columns.DATE] >= pd.Timestamp(start_date))
        ]

    config = [
        PerCaseTypeConfig(name=CaseTypes.CONFIRMED, dash_style=(1, 0), include=True),
        PerCaseTypeConfig(name=CaseTypes.DEATHS, dash_style=(1, 1), include=True),
    ]

    plot_size = (9, 9)

    return _plot_helper(
        df, style=style, filter_df_func=filter_func, config=config, plot_size=plot_size
    )


def plot_countries(
    df: pd.DataFrame,
    countries: List[str],
    *,
    style=None,
    start_date=None,
    include_recovered=False,
):
    def filter_func(df):
        return df[
            df[Columns.COUNTRY].isin(countries)
            & (df[Columns.DATE] >= pd.Timestamp(start_date))
            & (include_recovered | (df[Columns.CASE_TYPE] != CaseTypes.RECOVERED))
        ]

    config = [
        PerCaseTypeConfig(name=CaseTypes.CONFIRMED, dash_style=(1, 0), include=True),
        PerCaseTypeConfig(
            name=CaseTypes.RECOVERED,
            dash_style=(3, 3, 1, 3),
            include=include_recovered,
        ),
        PerCaseTypeConfig(name=CaseTypes.DEATHS, dash_style=(1, 1), include=True),
    ]

    plot_size = (12, 12)

    _plot_helper(
        df, style=style, filter_df_func=filter_func, config=config, plot_size=plot_size
    )


def get_country_cases_df(filepath: Path, *, case_type: str):
    case_type = case_type.title()

    df = pd.read_csv(filepath)
    df: pd.DataFrame
    df = df.melt(
        id_vars=[Columns.STATE, Columns.COUNTRY, Columns.LATITUDE, Columns.LONGITUDE],
        var_name=Columns.DATE,
        value_name=Columns.CASE_COUNT,
    )
    df[Columns.DATE] = pd.to_datetime(df[Columns.DATE])
    df[Columns.CASE_TYPE] = case_type

    return df


def get_world_cases_df(filepath: Path, *, case_type: str):
    df = get_country_cases_df(filepath, case_type=case_type)
    df = df.drop(columns=[Columns.LATITUDE, Columns.LONGITUDE])

    world_df = df.groupby([Columns.DATE, Columns.CASE_TYPE])[Columns.CASE_COUNT].sum()

    china_df = (
        df[df[Columns.COUNTRY] == "China"]
        .groupby([Columns.DATE, Columns.CASE_TYPE])[Columns.CASE_COUNT]
        .sum()
    )
    world_minus_china_df = world_df.sub(china_df)

    world_df = world_df.reset_index()
    world_minus_china_df = world_minus_china_df.reset_index()

    world_df[Columns.COUNTRY] = Strings.WORLD
    world_minus_china_df[Columns.COUNTRY] = Strings.WORLD_MINUS_CHINA

    df = pd.concat([world_df, world_minus_china_df], axis=0)

    return df


def join_dfs():
    dfs = []
    dfs: List[pd.DataFrame]

    for csv in DATA_PATH.glob("time_series_19*.csv"):
        case_type = csv.stem.replace("time_series_19-covid-", "")
        df = get_country_cases_df(csv, case_type=case_type)
        dfs.append(df)

    for csv in DATA_PATH.glob("time_series_covid19_*_global.csv"):
        case_type = csv.stem.replace("time_series_covid19_", "").replace("_global", "")
        df = get_world_cases_df(csv, case_type=case_type)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Remove cities in US (eg "New York, NY")
    df = df[~df[Columns.STATE].str.contains(",").fillna(False)]

    # Aggregate cities/province/states in select countries
    agg_country_df = (
        df[df[Columns.COUNTRY].isin(["China", "US"])]
        .groupby([Columns.DATE, Columns.CASE_TYPE, Columns.COUNTRY], as_index=False)
        .agg(
            {
                Columns.LATITUDE: "first",
                Columns.LONGITUDE: "first",
                Columns.CASE_COUNT: "sum",
            }
        )
    )

    df = pd.concat([df, agg_country_df], axis=0)

    # For countries other than the US, don't include their states/discontiguous regions
    # E.g., Gibraltar, Isle of Man, French Polynesia, etc
    df = df[
        (df[Columns.COUNTRY] == "US")
        | (df[Columns.STATE] == df[Columns.COUNTRY])  # France is like this, idk why
        | df[Columns.STATE].isna()
    ]

    # Minor cleanup
    df[Columns.COUNTRY] = df[Columns.COUNTRY].replace("Korea, South", "South Korea")
    df.loc[
        df[Columns.COUNTRY] == "Georgia", Columns.COUNTRY
    ] = "Georgia (country)"  # not the state

    # Use state as location name for states, else use country name
    df[Columns.LOCATION_NAME] = df[Columns.STATE].fillna(df[Columns.COUNTRY])

    # Hereafter df is sorted by date, which is helpful as it allows using .iloc[-1]
    # to get current (or most recent known) situation per location
    df = df.sort_values([Columns.LOCATION_NAME, Columns.DATE])
    return df


df = join_dfs()

df = df.groupby(Columns.LOCATION_NAME).filter(
    # Exclude US states (unless they meet certain criteria, see below) and
    # UK's discontiguous regions (e.g., Isle of Man, Gibraltar)
    lambda g: g[Columns.COUNTRY].iloc[0] not in ["US", "United Kingdom"]
    # Keep UK the country
    or (g.name == "United Kingdom")  # g.name is LOCATION_NAME
    or (
        # Keep top n US states by current number of confirmed cases
        g.loc[g[Columns.CASE_TYPE] == CaseTypes.CONFIRMED, Columns.CASE_COUNT].iloc[-1]
        >= df[(df[Columns.COUNTRY] == "US") & (df[Columns.LOCATION_NAME] != "US")]
        .groupby(Columns.LOCATION_NAME)
        .apply(
            lambda h: h.loc[
                h[Columns.CASE_TYPE] == CaseTypes.CONFIRMED, Columns.CASE_COUNT
            ].iloc[-1]
        )
        .nlargest(3)
        .iloc[-1]
    )
)


plot_countries(
    df,
    [
        # "China",
        "Italy",
        "Iran",
        "France",
        "Spain",
        "Germany",
        # "United Kingdom",
        "South Korea",
        "US",
    ],
    start_date="2020-2-20",
    include_recovered=False,
)

plot_world_and_china(df, start_date="2020-1-1")

plt.show()

# [[%%
