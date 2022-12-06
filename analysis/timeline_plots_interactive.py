from collections import namedtuple
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .process_txt import _clean_tweet, _get_bigrams


def group_daily(text_col: str = "bigram_lemmas"):
    """
    Aggregates tweet text on creation date.
    By defaults aggregates lemmas (inc bigrams)
    """
    df = pd.read_pickle("tweets_txt_processed-2022-12-06.pkl")
    df["created_at_day"] = df["created_at"].apply(
        lambda x: datetime(x.year, x.month, x.day)
    )
    df["created_at_month"] = (
        df["created_at"].dt.to_period("M").apply(lambda m: m.to_timestamp())
    )

    df = df.groupby("date").agg({text_col: "sum"}).reset_index()

    # Flatten list to str
    df["text"] = df[text_col].str.split()

    return df


def query_tokens_timeline(query, df):
    def count_occurence_df(docs, query, index):

        result_tup = namedtuple("Count", "date token count")
        results = []

        for doc, i in zip(docs, index):
            results.extend(
                result_tup(date=i, token=w, count=doc.count(w)) for w in query
            )
        return pd.DataFrame(results)

    query = [
        "cuckservative",
        "cuck",
        "gamergate",
        "sjw",
        "feminis",
        "feminaz",
        "white",
        "genocide",
        "Trump",
        "Normie",
        "shitlib",
        "shitposting",
        "alpha",
        "beta",
        "lol",
        "lul",
        "white",
        "suprema",
        "troll",
        "ridiculous",
        "maga",
    ]

    # Apply same preprocessing to query
    _, lemmas = _clean_tweet(" ".join(query))
    query = lemmas + _get_bigrams(lemmas)

    # Create dataframe used for plot
    plot_df = count_occurence_df(
        query, df["text"].tolist(), df["created_at_day"].tolist()
    )

    fig = go.Figure()

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
            fig.add_trace(go.Scatter(x=tmp["date"], y=tmp["count"], name=token))

    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
    )

    fig.write_html("plots/tokens_timeline.html", auto_open=True)
