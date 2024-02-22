from collections.abc import Iterable
from itertools import chain
import time
from typing import Callable
from urllib.parse import urlparse

from tqdm import tqdm
from tweepy import Client


def from_tweet_entities_get_source_url_factory(substring_to_found_in_url: str) -> Callable:
    def from_tweet_entities_get_source_url(quote_tweet_entitites):
        expanded_urls_in_tweet = [
            record["expanded_url"]
            for record in quote_tweet_entitites["urls"]
            if substring_to_found_in_url in record["expanded_url"]
        ]
        first_link = expanded_urls_in_tweet[0] if expanded_urls_in_tweet else "None"
        return f"{urlparse(first_link).netloc}{urlparse(first_link).path}"

    return from_tweet_entities_get_source_url


def get_tweets_metadata(client: Client, tweets_ids) -> list:
    tweets_ids = [tweet_id for tweet_id in tweets_ids if len(tweet_id) == 19]
    responses = []

    chunk_size = 100
    for i in tqdm(range(0, len(tweets_ids), chunk_size)):
        response = client.get_tweets(
            tweets_ids[i : i + chunk_size],
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
        )
        responses.append(response.data)
        time.sleep(0.2)

    responses = list(chain.from_iterable(item for item in responses if isinstance(item, Iterable)))
    return [response.data for response in responses]
