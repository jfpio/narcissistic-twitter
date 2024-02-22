import json
from pathlib import Path
import time
from typing import Callable, List

from dotenv import load_dotenv
from tweepy import Client, Paginator, Tweet

load_dotenv()


def save_tweets_to_file(
    tweets: list,  # list of downloaded tweets
    file_name: str,
    path_to_save: Path,
    silent: bool = False,
    logging_fun: Callable[[str], None] = print,
) -> None:
    path_to_save.mkdir(parents=True, exist_ok=True)
    tweets = [t for t in tweets if t]
    json_data = {"tweets": [t.data for t in tweets]}

    with open(file=path_to_save / file_name, mode="w") as outfile:
        json.dump(json_data, outfile, indent=4, sort_keys=True)

    if not silent:
        logging_fun(f"Successfully saved to {path_to_save}/{file_name}")


def download_tweet(tweet_id: int, client: Client) -> Tweet:
    tweet = client.get_tweet(
        id=tweet_id,
        tweet_fields=[
            "author_id",
            "context_annotations",
            "conversation_id",
            "entities",
            "lang",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
        ],
    )
    return tweet.data


def download_quotes_if_exists(tweet: Tweet, client: Client, limit: int) -> List[Tweet]:
    if tweet and tweet["public_metrics"]["quote_count"]:
        quotes = tweet["public_metrics"]["quote_count"]
        limit = min(limit, quotes) if limit != -1 else quotes
        return download_quotes(tweet["id"], client, limit)
    else:
        return []


def download_quotes(tweet_id: int, client: Client, quotes_num: int) -> List[Tweet]:
    tweets_chunk = 100
    tweets_gen = Paginator(
        client.get_quote_tweets,
        tweet_id,
        tweet_fields=[
            "author_id",
            "conversation_id",
            "lang",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "entities",
            "created_at",
            "attachments",
            "source",
        ],
        max_results=tweets_chunk,
    ).flatten(limit=quotes_num)

    tweets = []
    for count, tweet in enumerate(tweets_gen):
        if count % tweets_chunk == 0:
            time.sleep(1)
        # https://docs.tweepy.org/en/v4.10.0/faq.html#why-am-i-getting-rate-limited-so-quickly-when-using-client-search-all-tweets-with-paginator
        tweets.append(tweet)

    return tweets


def download_referenced_tweets(tweet: Tweet, client: Client) -> List[Tweet]:
    if not tweet:
        return []
    result = []
    if tweet.referenced_tweets:
        for ref_tweet in tweet.referenced_tweets:
            result.append(download_tweet(ref_tweet["data"]["id"], client))
    return result


def download_replies(tweet: dict, client: Client, limit: int) -> List[Tweet]:
    tweets_count = tweet["public_metrics"]["reply_count"]
    tweet_id = tweet["id"]
    query = f"in_reply_to_tweet_id:{tweet_id}"
    if limit != -1 and tweets_count > limit:
        print(f"Found {tweets_count} replies, but only {limit} will be downloaded.")
        tweets_count = limit

    tweets_chunk = 100
    tweets_gen = Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=[
            "author_id",
            "conversation_id",
            "lang",
            "public_metrics",
            "referenced_tweets",
            "reply_settings",
            "entities",
            "created_at",
            "attachments",
            "source",
        ],
        max_results=tweets_chunk,
    ).flatten(limit=tweets_count)

    tweets = []
    for count, tweet in enumerate(tweets_gen):
        if count % tweets_chunk == 0:
            time.sleep(1)
        # https://docs.tweepy.org/en/v4.10.0/faq.html#why-am-i-getting-rate-limited-so-quickly-when-using-client-search-all-tweets-with-paginator
        tweets.append(tweet)

    return tweets
