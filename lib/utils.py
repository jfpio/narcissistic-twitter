from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from tweepy import Client

T = TypeVar("T")


def flatten(list_of_lists: list[list[T]]) -> list[T]:
    return [item for sublist in list_of_lists for item in sublist]


def filter_nan(values: NDArray) -> NDArray:
    assert values.ndim == 1
    return values[~np.isnan(values)]


def get_tweets_count_to_query(
    client: Client, query: str, start_date: datetime, end_date: datetime, days_interval: timedelta, silent: bool = False
) -> int:
    assert isinstance(start_date, datetime) and isinstance(end_date, datetime) and isinstance(days_interval, timedelta)
    assert start_date < end_date

    if not silent:
        print(f"Getting tweets count for query: {query} from {start_date} to {end_date} with interval {days_interval}.")

    responses = []
    for start_time, end_time in generate_date_intervals(start_date, end_date, days_interval):
        count_response = client.get_recent_tweets_count(
            query=query, start_time=start_time.isoformat(), end_time=end_time.isoformat(), granularity="day"
        )  # TODO
        responses.append(count_response)

    tweets_count = 0
    for response in responses:
        for result in response.data:
            tweets_count += result["tweet_count"]

    if not silent:
        print(f"Found {tweets_count} tweets.")

    return tweets_count


def generate_date_intervals(
    start_date: datetime, end_date: datetime, interval: timedelta
) -> Generator[tuple[datetime, datetime], None, None]:
    interval_start = start_date
    interval_end = start_date + interval

    while interval_end < end_date:
        yield interval_start, interval_end
        interval_start = interval_end
        interval_end += interval
    yield interval_start, end_date
