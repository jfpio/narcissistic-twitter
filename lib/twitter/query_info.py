from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Optional

from tqdm import tqdm
from tweepy import Client, Paginator

from lib.utils import flatten, generate_date_intervals, get_tweets_count_to_query


class QueryInfo:
    def __init__(
        self,
        client: Client,
        query: str,
        start_date: datetime,
        end_date: datetime,
        file_name: str,
        path_to_save: Optional[str] = None,
        tweets_chunk: Optional[int] = None,
        days_interval: Optional[int] = None,
        tweet_fields: Optional[list[str]] = None,
        user_fields: Optional[list[str]] = None,
        max_tries: Optional[int] = None,
        save_to_temporary_files: bool = True,
        max_tweets: Optional[int] = None,
        sort_order: Optional[str] = None,
    ):
        self.client = client
        self.query = query
        self.start_date = start_date
        self.end_date = end_date
        self.file_name = file_name
        default_path_to_save = "/dysk1/approx/media_monitoring/pro_ukraine/tweety-ukraina/"
        if path_to_save is None:
            path_to_save = default_path_to_save
        self.path_to_save = path_to_save
        if tweets_chunk is None:
            tweets_chunk = 100
        self.tweets_chunk = tweets_chunk
        if days_interval is None:
            days_interval = 28
        self.days_interval = timedelta(days=days_interval)
        if tweet_fields is None:
            tweet_fields = [
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
            ]
        self.tweet_fields = tweet_fields
        if user_fields is None:
            user_fields = ["id", "name", "created_at", "description"]
        self.user_fields = user_fields
        self.save_to_temporary_files = save_to_temporary_files
        if max_tries is None:
            max_tries = 4
        self.max_tries = max_tries
        if max_tweets is None:
            max_tweets = -1
        self.max_tweets = max_tweets
        self.sort_order = sort_order

        assert isinstance(client, Client)
        assert isinstance(query, str)
        assert isinstance(start_date, datetime) and isinstance(end_date, datetime) and start_date < end_date
        assert isinstance(file_name, str)
        assert isinstance(path_to_save, str)
        assert isinstance(tweets_chunk, int) and tweets_chunk > 0
        assert isinstance(days_interval, int) and days_interval > 0
        assert isinstance(tweet_fields, list) and all(isinstance(field, str) for field in tweet_fields)
        assert isinstance(user_fields, list) and all(isinstance(field, str) for field in user_fields)
        if "context_annotations" in tweet_fields:
            assert tweets_chunk <= 100, "Set tweets_chunk to at most 100 when using context annotations"
            # https://twittercommunity.com/t/max-results-and-context-annotations/156427
        assert isinstance(save_to_temporary_files, bool)
        assert isinstance(max_tries, int) and max_tries > 0

        self.test_path_to_save()

    def download_tweets(self) -> list:  # list of tweets
        def try_get_tweets(start: datetime, end: datetime, tries_count: int = 0) -> None:
            tweets_number = get_tweets_count_to_query(self.client, self.query, start, end, self.days_interval)
            try:
                tweets = self._get_tweets(start, end, tweets_number)
                monthly_tweets.append(tweets)
                if self.save_to_temporary_files:
                    self._save_to_temp_file(flatten(monthly_tweets), end)
            except Exception as e:
                tries_count += 1
                print(f"Failed to get tweets for interval {start} - {end} after {tries_count} tries.")
                if tries_count < self.max_tries:
                    failed.append((start, end, tries_count))
                else:
                    print(f"Max tries. Skipping the interval {start} - {end}.")
                print(e)

        intervals = generate_date_intervals(self.start_date, self.end_date, self.days_interval)
        monthly_tweets = []
        failed = []

        for start, end in intervals:
            try_get_tweets(start, end)

        while failed:
            start, end, tries_count = failed.pop()
            try_get_tweets(start, end, tries_count)

        return flatten(monthly_tweets)

    def save_to_file(
        self,
        tweets: list,  # list of downloaded tweets
        file_name: Optional[str] = None,
        end_date: Optional[datetime] = None,
        silent: bool = False,
    ) -> None:
        path_to_save = Path(self.path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)
        if file_name is None:
            file_name = f"{self.file_name}-{datetime.now().isoformat()}"
        file_name += ".json"

        start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        json_data = {
            "query": self.query,
            "start_time": start_date.astimezone().isoformat(),
            "end_time": end_date.astimezone().isoformat(),
            "tweets": list([tweet.data for tweet in tweets]),
        }

        with open(file=path_to_save / file_name, mode="w") as outfile:
            json.dump(json_data, outfile, indent=4, sort_keys=True)

        if not silent:
            print(f"Successfully saved to {self.path_to_save}/{file_name}")

    def cleanup(self) -> None:
        if self.save_to_temporary_files:
            self._remove_temp_file()

    def get_tweets_count(self, silent: bool = False) -> int:
        return get_tweets_count_to_query(
            self.client,
            self.query,
            self.start_date,
            self.end_date,
            self.days_interval,
            silent,
        )

    def test_path_to_save(self) -> None:
        try:
            self._save_to_temp_file([], end_date=self.start_date)
            self._remove_temp_file()
        except Exception as e:
            print(f"Saving to temporary file failed. Make sure that the path {self.path_to_save} is correct.")
            raise e

    def _create_generator(self, start_date: datetime, end_date: datetime, tweets_number: int) -> Paginator:
        assert isinstance(start_date, datetime) and isinstance(end_date, datetime) and start_date < end_date

        return Paginator(
            self.client.search_recent_tweets,  # TODO
            query=self.query,
            tweet_fields=self.tweet_fields,
            user_fields=self.user_fields,
            start_time=start_date.isoformat(),
            end_time=end_date.isoformat(),
            max_results=self.tweets_chunk,
            sort_order=self.sort_order,
        ).flatten(tweets_number)

    def _get_tweets(self, start_date: datetime, end_date: datetime, tweets_number: int) -> list:  # list of tweets
        if self.max_tweets != -1:
            tweets_number = min(tweets_number, self.max_tweets)
        tweets_gen = self._create_generator(start_date, end_date, tweets_number)
        tweets = []

        for count, tweet in tqdm(enumerate(tweets_gen), total=tweets_number):
            if count % self.tweets_chunk == 0:
                time.sleep(1)
            # https://docs.tweepy.org/en/v4.10.0/faq.html#why-am-i-getting-rate-limited-so-quickly-when-using-client-search-all-tweets-with-paginator
            tweets.append(tweet)
        return tweets

    def _get_temp_file_name(self) -> str:
        return "temp_" + self.file_name

    def _save_to_temp_file(self, tweets: list, end_date: Optional[datetime] = None) -> None:
        self.save_to_file(tweets, self._get_temp_file_name(), end_date, silent=True)

    def _remove_file(self, file_name: Optional[str] = None) -> None:
        if file_name is None:
            file_name = self.file_name
        file_name += ".json"

        path_to_save = Path(self.path_to_save)
        file_path = path_to_save / file_name
        if file_path.exists():
            file_path.unlink()

    def _remove_temp_file(self) -> None:
        self._remove_file(self._get_temp_file_name())
