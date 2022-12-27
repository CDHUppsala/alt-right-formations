import matplotlib.pyplot as plt
import pandas as pd


def prep_data():
    df = pd.read_pickle("tweets_txt_processed.pkl")

    df["month_year"] = (
        df["tweet_created_at"].dt.to_period("M").apply(lambda m: m.to_timestamp())
    )
    return (
        df.groupby("month_year")
        .reset_index()[["date_only", "tweet_id"]]
        .rename(columns={"tweet_id": "count"})
    )


def plot_timeseries(df):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(df["month_year"], df["count"], "k")
    ax.set_title("Frequency of Alt-Right Tweets per Month")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    fig.savefig("altright-timeline.pdf")
