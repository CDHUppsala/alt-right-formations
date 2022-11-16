import argparse
import logging
import re
import string
from multiprocessing import Pool
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
)

# Instantiate the stemmer, lemmatizer and tokenizer objects
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer()

# Get stopwords
stopwords = list(nltk.corpus.stopwords.words("english"))


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to pickle with Tweets")
    parser.add_argument("-n", "--njobs", type=int, default=4)
    return parser.parse_args()


def _clean_tweet(tweet: str) -> Tuple[str, str]:
    # Lowercasing words
    tweet = tweet.lower()

    tweet = re.sub(r"…", "", tweet)

    # Removing '&amp' which was found to be common
    tweet = re.sub(r"&amp", "", tweet)

    # Replace other instances of "&" with "and"
    tweet = re.sub(r"&", "and", tweet)

    # Removing mentions
    tweet = re.sub(r"@\w+ ", "", tweet)

    # Removing 'RT' and 'via'
    tweet = re.sub(r"(^rt|^via)((?:\b\W*@\w+)+): ", "", tweet)

    # Removing emojis
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )

    tweet = re.sub(EMOJI_PATTERN, "", tweet)

    # Removing punctuation
    my_punctuation = string.punctuation.replace("#", "")
    my_punctuation = my_punctuation.replace("-", "")

    tweet = tweet.translate(str.maketrans("", "", my_punctuation))
    tweet = re.sub(
        r" - ", "", tweet
    )  # removing dash lines bounded by whitespace (and therefore not part of a word)
    tweet = re.sub(
        r"[’“”—,!]", "", tweet
    )  # removing punctuation that is not captured by string.punctuation

    # Removing odd special characters
    tweet = re.sub(r"[┻┃━┳┓┏┛┗]", "", tweet)
    tweet = re.sub(r"\u202F|\u2069|\u200d|\u2066", "", tweet)

    # Removing URLs
    tweet = re.sub(r"http\S+", "", tweet)

    # Removing numbers
    tweet = re.sub(r"[0-9]", "", tweet)

    # Removing separators and superfluous whitespace
    tweet = tweet.strip()
    tweet = re.sub(r" +", " ", tweet)

    # Tokenizing
    tokens = tokenizer.tokenize(tweet)

    # Removing stopwords
    tokens = [w for w in tokens if w not in stopwords]

    # Lemmatize or stem the tok
    stems = [stemmer.stem(w) for w in tokens]
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(stems), " ".join(lemmas)


def process_text(tweets: list, n_jobs: int = 4):

    with Pool(n_jobs) as p:
        # stems, lemmas =
        return tqdm(
            zip(*p.imap(_clean_tweet, tweets)),
            total=len(tweets),
        )


def main():

    args = parse_arguments()
    logging.info(f"Reading dataset: {args.input}")
    df = pd.read_pickle(args.input)

    logging.info(f"Processing {df.shape[0]} tweets across {args.njobs} workers")
    df["stems"], df["lemmas"] = process_text(df["text"].tolist(), args.njobs)

    pkl_path = "tweets_txt_processed.pkl"
    logging.info(f"Saving to {pkl_path}")
    df.to_pickle(pkl_path)


if __name__ == "__main__":
    main()
