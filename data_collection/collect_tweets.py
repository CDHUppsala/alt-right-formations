import yaml
import tweepy
import argparse
import tqdm
import logging
import pathlib
from datetime import datetime
from dataclasses import dataclass

# Setup logging
logpath = pathlib.Path(f"collect_logs/{datetime.now().strftime('%Y-%m-%d_%H:%M')}.log")
if not logpath.parent.exists():
    logpath.parent.mkdir(parents=True)
logging.basicConfig(
    handlers=[
        logging.FileHandler(filename=logpath, mode="a"),
        logging.StreamHandler(),
    ],
    format="%(asctime)s | %(levelname)s: %(message)s",
    level=logging.INFO,
)


def parse_config():
    with open("config.yaml", "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            raise


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--token", type=str, help="Twitter bearer token.")
    parser.add_argument(
        "-n", "--ntweets", type=int, default=100, help="Number of Tweets to collect."
    )
    return parser.parse_args()


@dataclass
class Tweet:
    tweet_id:int
    author_id:int
    tweet_created_at:datetime
    text:str
    possibly_sensitive:bool
    lang:str
    geo:str
    retweet_count:int
    reply_count:int
    like_count:int
    quote_count:int
    following_count:int
    tweet_count:int
    listed_count:int
    username:str
    name:str
    description:str
    account_created_at:datetime
    location:str
    verified:bool

class TweetCollector:

    def __init__(self, bearer_token: str):
        
        self.client = tweepy.Client(bearer_token=bearer_token, 
                                    wait_on_rate_limit=True)
    
    def _parse_response(response:tweepy.Response):





def collect_tweets(
    query: str,
    start_time: datetime,
    end_time: datetime,
    fields:dict,
    n_tweets: int,
    bearer: str,
) -> None:

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)

    response = client.search_all_tweets(
        query=query,
        start_time=start_time,
        end_time=end_time,
        tweet_fields = fields.get("tweet_fields"),
        user_fields=fields.get("user_fields"),
        place_fields=fields.get("place_fields"),
        max_tweets = 500
    )

    for tweet in paginator.flatten(limit=n_tweets):




def main():

    config = parse_config()
    args = parse_arguments()


if __name__ == "__main__":
    main()
