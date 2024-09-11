import json
import os

import requests

bearer_token = os.environ["BEARER_TOKEN"]

search_url = "https://api.twitter.com/2/tweets/search/recent"

# Optional params: start_time,end_time,since_id,until_id,max_results,next_token,
# expansions,tweet.fields,media.fields,poll.fields,place.fields,user.fields
query_params = {"query": "(from:twitterdev -is:retweet) OR #twitterdev", "tweet.fields": "author_id"}


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r


def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def main():
    json_response = connect_to_endpoint(search_url, query_params)
    print(json.dumps(json_response, indent=4, sort_keys=True))
    # Save json_response to a file in data raw folder
    with open("data/raw/twitter_recent_search.json", "a") as f:
        json.dump(json_response, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
