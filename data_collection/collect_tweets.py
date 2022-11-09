import argparse
import logging
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import tweepy
import yaml
from tqdm import tqdm
from tweepy.errors import TooManyRequests, TwitterServerError

# Setup logging
log_path = pathlib.Path(f"collect_logs/{datetime.now().strftime('%Y-%m-%d_%H:%M')}.log")
if not log_path.parent.exists():
    log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    handlers=[
        logging.FileHandler(filename=log_path, mode="a"),
        logging.StreamHandler(),
    ],
    format="%(asctime)s | %(levelname)s: %(message)s",
    level=logging.INFO,
)


def parse_config():
    with open("../config.yaml", "r") as stream:
        return yaml.safe_load(stream).get("data_collection")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--token", type=str, help="Twitter bearer token.")
    parser.add_argument(
        "-n", "--ntweets", type=int, default=100, help="Number of Tweets to collect."
    )
    parser.add_argument("--next", type=str, help="Next token in paginator")
    return parser.parse_args()


@dataclass
class Tweet:
    tweet_id: int
    author_id: int
    tweet_created_at: datetime
    text: str
    possibly_sensitive: bool
    lang: str
    geo: str
    referenced_tweets: dict
    source: str
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int
    following_count: int
    tweet_count: int
    listed_count: int
    username: str
    name: str
    description: str
    account_created_at: datetime
    location: str
    verified: bool


def _parse_response(response: tweepy.Response) -> List[Tweet]:
    """
    Parses Tweet and User data.
    """

    user_lookup = {user.id: user for user in response.includes["users"]}

    tweets = []
    for tweet in response.data:
        try:
            tweets.append(
                Tweet(
                    tweet_id=tweet.id,
                    author_id=tweet.author_id,
                    tweet_created_at=tweet.created_at,
                    text=tweet.text,
                    possibly_sensitive=tweet.possibly_sensitive,
                    lang=tweet.lang,
                    geo=tweet.geo,
                    referenced_tweets=tweet.data.get("referenced_tweets"),
                    source=tweet.source,
                    retweet_count=tweet.public_metrics["retweet_count"],
                    reply_count=tweet.public_metrics["reply_count"],
                    like_count=tweet.public_metrics["like_count"],
                    quote_count=tweet.public_metrics["quote_count"],
                    # user stuff
                    following_count=user_lookup[tweet.author_id].public_metrics[
                        "following_count"
                    ],
                    tweet_count=user_lookup[tweet.author_id].public_metrics[
                        "tweet_count"
                    ],
                    listed_count=user_lookup[tweet.author_id].public_metrics[
                        "listed_count"
                    ],
                    username=user_lookup[tweet.author_id].username,
                    name=user_lookup[tweet.author_id].name,
                    description=user_lookup[tweet.author_id].description,
                    account_created_at=user_lookup[tweet.author_id].created_at,
                    location=user_lookup[tweet.author_id].location,
                    verified=user_lookup[tweet.author_id].verified,
                )
            )
        except KeyError:
            logging.error(f"Tweet ID: {tweet.id} has no user info. Skipping...")

    return tweets


def _request_tweets(client, query, start_time, end_time, fields, next_token):

    tries = 10
    for attempt in range(tries):
        try:
            response = client.search_all_tweets(
                query=query,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=fields.get("tweet_fields"),
                user_fields=fields.get("user_fields"),
                max_results=500,
                expansions=["author_id", "referenced_tweets.id"],
                next_token=next_token,
            )

        except (TwitterServerError, TooManyRequests) as e:
            logging.error(f"Request failed, sleeping for 10 sec. Attempt: {attempt}.")
            logging.error(e)
            time.sleep(10)
            continue

        else:
            break
    else:
        logging.error(f"Request failed after {tries} attempts. Continuing...")
        return

    return response


def collect_tweets(
    query: str,
    start_time: datetime,
    end_time: datetime,
    fields: dict,
    n_tweets: int,
    bearer_token: str,
    next_token: Optional[str] = None,
) -> None:

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    logging.info(f"Client connected. Starting to collect {n_tweets} tweets.")

    # Get initial query
    response = _request_tweets(
        client=client,
        query=query,
        start_time=start_time,
        end_time=end_time,
        fields=fields,
        next_token=next_token,
    )
    if not response:
        raise Exception("First request failed. Stopping...")

    # Save to Csv
    csv_path = pathlib.Path(f"data/{datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv")
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(_parse_response(response))
    df.to_csv(csv_path, header=True, index=False)

    # setup progressbar
    n_total = len(response.data)
    pbar = tqdm(total=n_tweets)
    pbar.set_description("Tweets collected: ")
    pbar.update(n_total)

    next_token = response.meta["next_token"]

    while n_total < n_tweets:

        time.sleep(1)

        # Get initial query
        response = _request_tweets(
            client=client,
            query=query,
            start_time=start_time,
            end_time=end_time,
            fields=fields,
            next_token=next_token,
        )

        if not response:
            continue

        df = pd.DataFrame(_parse_response(response))
        df.to_csv(csv_path, header=False, index=False, mode="a")

        # Update token and write to file
        next_token = response.meta["next_token"]
        with open("next_token.txt", "w") as t:
            t.write(next_token)

        n_new = len(response.data)
        pbar.update(n_new)
        n_total += n_new

    logging.info("Data collection finished.")


def main():

    config = parse_config()
    args = parse_arguments()

    collect_tweets(
        query=config["query"],
        start_time=config["start_time"],
        end_time=config["end_time"],
        fields=config["fields"],
        n_tweets=args.ntweets,
        bearer_token=args.token,
        next_token=args.next,
    )


if __name__ == "__main__":
    main()
