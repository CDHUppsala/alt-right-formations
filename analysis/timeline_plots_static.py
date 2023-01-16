from collections import namedtuple
from datetime import datetime
import re

import pandas as pd
import matplotlib.pyplot as plt
import yaml
import matplotlib.dates as mdates


# Read in tokens to plot
with open("queries.yaml", "r") as stream:
    queries = yaml.safe_load(stream)


def prepare_df(df, text_col="lemmas_bigrams"):
    """Prepare the dataframe for plotting"""
    # Convert lemmas to string
    df[text_col] = df[text_col].apply(lambda x: " ".join(x))
    # Drop ambivalent tweet types
    # Group by day and tweet type. Then join the text and count the number of tweets
    df = (
        df.query("tweet_type != 'ambivalent'")
        .groupby(["created_at_day", "tweet_type"])[text_col]
        .agg([" ".join, "count"])
        .rename(columns={"join": "text", "count": "tweet_count"})
        .reset_index()
    )

    def reindex_by_date(df, dates):
        return df.reindex(dates, fill_value=0)

    df.set_index("created_at_day", inplace=True)
    dates = pd.date_range(df.index.min(), df.index.max())

    # ffill tweet_type for missing dates grouped by tweet_type
    tweet_types = (
        df.groupby("tweet_type")
        .apply(lambda x: x.reindex(dates, method="ffill"))
        .index.get_level_values(0)
    )

    df = (
        df.groupby("tweet_type")
        .apply(lambda x: reindex_by_date(x, dates))
        .reset_index(0, drop=True)
    )

    df["tweet_type"] = tweet_types

    # Fill text column with empty string where tweet_count is 0
    df.loc[df["tweet_count"] == 0, "text"] = ""

    df["total"] = df.groupby(df.index)["tweet_count"].sum()
    df["frac"] = (df["tweet_count"] / df["total"]).fillna(0)

    return df


def plot_tweet_count(
    df,
    start_date="2016-01-01",
    end_date="2021-01-31",
    total=True,
    tweet_type="original",
    title="Daily Tweet Count",
    fname="daily_tweet_count.pdf",
):
    # with plt.style.context('science'):

    mask = (df.index >= start_date) & (df.index <= end_date)

    df = df.loc[mask]
    df = df.loc[df["tweet_type"] == tweet_type]

    y = df["total"] if total else df["tweet_count"]
    # y = df["tweet_count"]
    # y2 = df["total"]  -df["tweet_count"]

    year_month_formatter = mdates.DateFormatter(
        "%Y-%m"
    )  # four digits for year, two for month

    fig, ax = plt.subplots(dpi=500, figsize=(6, 3))
    ax.fill_between(df.index, y, color=plt.cm.Set2.colors[0])
    # ax.plot(df.index, y, linewidth = 0.6)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    ax.set_ylabel("Count")
    ax.set_xlabel("Date")

    ax.set_title(title)

    fig.savefig(f"plots/static/{fname}")


def stacked_area_plot(
    df,
    start_date="2016-01-01",
    end_date="2021-01-31",
    agg_lvl="W",
    title="Propotion of Tweet Types",
    fname="tweet_type_propotion_week.pdf",
):

    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]

    df["x"] = (
        df.index.to_series().dt.to_period(agg_lvl).apply(lambda x: x.to_timestamp())
    )
    df = df.groupby(["x", "tweet_type"]).mean().reset_index()

    y = [
        df.loc[df["tweet_type"] == t].frac * 100
        for t in ("retweeted", "original", "replied_to", "quoted")
    ]

    fig, ax = plt.subplots(dpi=500, figsize=(6, 3))
    labels = ["Retweet", "Original", "Reply", "Quote"]
    ax.stackplot(
        df["x"].unique(), y, labels=labels, colors=plt.cm.Set2.colors[: len(labels)]
    )  # , colors = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"])

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    # ax.legend(loc='lower right', fontsize = 8)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.set_ylabel("Percent (%)")
    ax.set_xlabel("Date")
    ax.margins(0, 0)
    fig.tight_layout()

    ax.set_title(title)
    fig.savefig(f"plots/static/{fname}")

def count_occurance(text:str, query) -> int:
    c = 0
    for name in query:
        c += len(re.findall(name, text))
    return c

def plot_sets_count(
    df,
    retweets=False,
    start_date="2016-01-01",
    end_date="2021-01-31",
    title="Alt-Right vs. Alt-Lite Tweet Count",
    fname="alt_right_vs_alt_lite.pdf",
):
    df = df.copy()
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "created_at_day"}, inplace=True)

    for s, q in queries["altright_vs_altlite"].items():
        df[s] = df["text"].apply(lambda x: count_occurance(x, q))

    if not retweets:
        df = df.loc[df["tweet_type"] == "original"]
    else:
        df = df.groupby(["created_at_day", "tweet_type"]).sum().reset_index()

    y_max = max(df["altright"].max(), df["altlite"].max())

    cmap = plt.cm.Set2.colors
    fig, axs = plt.subplots(dpi=500, figsize=(7, 4), nrows=2)

    for i, y, label in enumerate(
        zip(("altright", "altlite"), ("Alt-Right", "Alt-Lite"))
    ):
        axs[i].plot(
            df["created_at_day"], df[y], linewidth=0.6, color=cmap[i], label=label
        )
        axs[i].set_ylabel("Count")
        axs[i].set_xlabel("Date")
        axs[i].set_ylim(0, y_max)
        axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        axs[i].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[i].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(axs[i].xaxis.get_major_locator())
        )
        axs[i].legend()

    axs[0].set_title(title)
    fig.tight_layout()
    fig.savefig(f"plots/static/{fname}")


def main():
    df = pd.read_parquet(
        "2022-11-13_processed.parquet",
        columns=["created_at_day", "tweet_type", "lemmas_bigrams"],
        engine="fastparquet",
    )

    df = prepare_df(df)

    plot_tweet_count(
        df,
        start_date="2016-01-01",
        end_date="2021-01-31",
        total=False,
        tweet_type="original",
        title="Daily Count of Original Tweets",
        fname="daily_tweet_count.pdf",
    )

    plot_tweet_count(
        df,
        total=True,
        title="Daily Tweet Count",
        fname="daily_tweet_count_total.pdf",
    )

    stacked_area_plot(
        df,
        agg_lvl="W",
        title="Propotion of Tweet Types",
        fname="tweet_type_propotion_week.pdf",
    )

    stacked_area_plot(
        df,
        agg_lvl="D",
        title="Propotion of Tweet Types",
        fname="tweet_type_propotion_day.pdf",
    )

    plot_sets_count(
        df,
        retweets=False,
        title="Alt-Right vs. Alt-Lite",
        fname="alt_right_vs_alt_lite.pdf",
    )

    plot_sets_count(
        df,
        retweets=True,
        title="Alt-Right vs. Alt-Lite",
        fname="alt_right_vs_alt_lite_retweets.pdf",
    )


if __name__ == "__main__":
    main()
