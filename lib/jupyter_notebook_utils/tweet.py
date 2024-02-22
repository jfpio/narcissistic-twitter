import requests


class Tweet(object):
    """Display a tweet as iframe in jupyter notebook"""

    def __init__(self, tweet_id):
        api = f"https://publish.twitter.com/oembed?url=https://twitter.com/anybody/status/{tweet_id}"
        response = requests.get(api)
        self.text = response.json()["html"]

    def _repr_html_(self):
        return self.text
