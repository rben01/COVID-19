# %%
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display  # noqa
from matplotlib.ticker import ScalarFormatter

ROOT_PATH = Path("..")
DATA_PATH = ROOT_PATH / "csse_covid_19_data" / "csse_covid_19_time_series"

STATE_COL = "Province/State"
COUNTRY_COL = "Country/Region"
DATE_COL = "Date"
CASES_COL = "Cases"
CASE_TYPE_COL = "Case Type"


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
    return df


df = join_dfs()

df = df[
    (df[COUNTRY_COL] == "US")
    & (~df[STATE_COL].str.contains(",").fillna(False))
    & (df[DATE_COL] >= pd.to_datetime("2020-03-10"))
]
# akjdh = (
#     df.groupby(STATE_COL)
#     .apply(lambda h: h.loc[h[CASE_TYPE_COL] == "Confirmed", CASES_COL].max())
#     .nlargest(10)
#     .iloc[-1]
# )
# display(akjdh)
df = df.groupby(STATE_COL).filter(
    lambda g: g.loc[g[CASE_TYPE_COL] == "Confirmed", CASES_COL].max()
    >= df.groupby(STATE_COL)
    .apply(lambda h: h.loc[h[CASE_TYPE_COL] == "Confirmed", CASES_COL].max())
    .nlargest(8)
    .iloc[-1]
)
# display(df)
# sns.set()
g = sns.lineplot(
    x=DATE_COL,
    y=CASES_COL,
    data=df,
    hue=STATE_COL,
    style=CASE_TYPE_COL,
    style_order=["Confirmed", "Recovered", "Deaths"],
)
plt.xticks(rotation=45)
plt.yscale("log", basey=8, nonposy="mask")
plt.ylim(ymin=1)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.grid(b=True, which="both", axis="y")
plt.grid(b=True, which="both", axis="x")
plt.gcf().set_size_inches((10, 12))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# %%
