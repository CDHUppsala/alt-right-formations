from collections import namedtuple
from datetime import datetime
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yaml
import logging

logging.basicConfig(level=logging.INFO)


# Read in tokens to plot
with open("queries.yaml", "r") as stream:
    queries = yaml.safe_load(stream)


def create_dfs(text_col: str = "lemmas_bigrams"):
    """
    Aggregates tweet text on creation date.
    By defaults aggregates lemmas (inc bigrams)
    """
    fname = "2022-11-13_processed.parquet"
    logging.info(f"Starting to read {fname}")
    df = pd.read_parquet(
        "2022-11-13_processed.parquet",
        columns=["tweet_created_at", "referenced_tweets", text_col],
        engine="fastparquet",
    )

    logging.info("Aggregating on day and month")

    df["created_at_day"] = df["tweet_created_at"].apply(
        lambda x: datetime(x.year, x.month, x.day)
    )
    df["created_at_month"] = (
        df["created_at_day"].dt.to_period("M").apply(lambda m: m.to_timestamp())
    )

    # Check if retweet/reply/quote
    df["retweet"] = df["referenced_tweets"].apply(
        lambda x: True if x != "nan" else False
    )

    # Flatten list to str
    df["text"] = df[text_col].apply(" ".join)
    df.drop(columns=[text_col, "referenced_tweets", "tweet_created_at"], inplace=True)

    # Group df by the day
    df_daily = (
        df.groupby("created_at_day")["text"]
        .agg([" ".join, "count"])
        .rename(columns={"join": "text", "count": "tweet_count"})
        .reset_index()
    )
    df_daily_noretweet = (
        df.query("retweet==False")
        .groupby("created_at_day")["text"]
        .agg([" ".join, "count"])
        .rename(columns={"join": "text", "count": "tweet_count"})
        .reset_index()
    )


    # Group df by both year and month
    df_month = df.groupby("created_at_month")["text"].apply(" ".join).reset_index()
    df_month_noretweet = (
        df.query("retweet==False")
        .groupby("created_at_month")["text"]
        .apply(" ".join)
        .reset_index()
    )
    # Freeup some memory
    del df

    logging.info("Reading data finished")

    return df_daily, df_daily_noretweet, df_month, df_month_noretweet


def tweet_freq_timeline(df, filename):
    """
    Plots the daily count of tweets.
    """

    fig = px.line(
        df,
        x="created_at_day",
        y="tweet_count",
        title="Daily Counts of Alt-Right Tweets",
    )

    fig.update_yaxes(fixedrange=False, autorange=True)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )

    fig.write_html(f"plots/{filename}")


def query_tokens_timeline(df, query, filename="tokens_timeline.html"):
    """
    Creates an interactive figure that displays
    the daily frequency of a provided query.
    """

    def count_occurence_df(docs, query, index):
        """
        Helper function for creating a dataframe of token counts.
        """

        result_tup = namedtuple("Count", "created_at_day token count")
        results = []

        for doc, i in zip(docs, index):
            results.extend(
                result_tup(created_at_day=i, token=w, count=len(re.findall(w, doc)))
                for w in query
            )
        return pd.DataFrame(results)

    # Create dataframe used for plot
    plot_df = count_occurence_df(
        df["text"].tolist(), query, df["created_at_day"].tolist()
    )

    layout = {"yaxis": {"autorange": True, "fixedrange": False}}

    fig = go.Figure(layout=layout)

    unique_tokens = plot_df.token.unique()
    n_tokens = len(unique_tokens)
    mask = [False] * n_tokens

    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * n_tokens}, {"title": "All", "showlegend": True}],
        )
    ]

    for i, token in enumerate(unique_tokens):
        tmp = plot_df.loc[plot_df["token"] == token]
        tmp_mask = mask.copy()
        tmp_mask[i] = True
        if tmp["count"].sum() > 200:
            fig.add_trace(
                go.Scatter(x=tmp["created_at_day"], y=tmp["count"], name=token)
            )

    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )

    fig.write_html(f"plots/{filename}")


def sets_timeline(df, name_sets):

    for name, tokens in name_sets.items():

        tokens_regex = "|".join(tokens)
        df[name] = df["text"].str.count(tokens_regex)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["created_at_day"], y=df["set1"], name="set 1"))
    fig.add_trace(go.Scatter(x=df["created_at_day"], y=df["set2"], name="set 2"))
    fig.add_trace(go.Scatter(x=df["created_at_day"], y=df["set3"], name="set 3"))

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )

    fig.update_layout(autosize=False, width=1600, height=800)

    fig.write_html("plots/sets_timeline.html")


def create_barplot(df: pd.DataFrame, n_largest, filename):

    plot_data = {}
    for c in df.columns:
        plot_data[c] = df[c].nlargest(n_largest).to_dict()

    fig = go.Figure()

    for m, d in plot_data.items():
        # print(m.strftime("%Y-%m-%d"))
        fig.add_trace(
            go.Bar(
                y=list(d.keys()),
                x=list(d.values()),
                name=m.strftime("%m/%d/%Y"),
                orientation="h",
            )
        )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        font=dict(size=8),
        autosize=False,
        width=1000,
        height=1200,
    )

    fig.write_html(f"plots/{filename}")


def top_tokens_barplot(df, filename):
    """
    Creates a barplot of the n most important tokens
    according to term frequency and tfidf each month.
    """

    stopwords = [
        "right",
        "altright",
        "_alt",
        "alt",
        "alt-right",
        "alt_right",
        "right",
        "alternative",
        "alternative_right",
        "altright_",
    ]

    tfidf_vec = TfidfVectorizer(min_df=30, max_df=0.99, stop_words=stopwords)
    tfidf = tfidf_vec.fit_transform(df["text"])

    tfidf = pd.DataFrame(
        tfidf.toarray().T,
        columns=df["created_at_month"].tolist(),
        index=tfidf_vec.get_feature_names_out(),
    )

    tf_vec = CountVectorizer(min_df=30, max_df=0.99, stop_words=stopwords)
    tf = tf_vec.fit_transform(df["text"])

    tf = pd.DataFrame(
        tf.toarray().T,
        columns=df["created_at_month"].tolist(),
        index=tf_vec.get_feature_names_out(),
    )

    create_barplot(tf, filename=f"tf_{filename}", n_largest=100)
    create_barplot(tfidf, filename=f"tfidf_{filename}.html", n_largest=100)


def main():

    # Create datasets for plots
    df_day, df_day_noretweet, df_month, df_month_noretweet = create_dfs()

    tweet_freq_timeline(df_day, "tweet_freq_timeline.html")
    tweet_freq_timeline(df_day_noretweet, "tweet_freq_timeline_noretweets.html")

    top_tokens_barplot(df_month, filename="barplot.html")
    top_tokens_barplot(df_month_noretweet, filename="barplot_noretweets.html")

    for name, query in queries.get("sets").items():
        query_tokens_timeline(df_day, query, filename=f"{name}_timeline.html")
        query_tokens_timeline(
            df_day_noretweet, query, filename=f"{name}_timeline_noretweets.html"
        )

    # query_tokens_timeline(df_day, names_query, filename="names_timeline.html")
    # sets_timeline(df_day, name_sets)


if __name__ == "__main__":
    main()
