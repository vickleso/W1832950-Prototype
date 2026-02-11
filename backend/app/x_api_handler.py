import requests
import re
import os
import traceback
from dotenv import load_dotenv

load_dotenv()

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

class XAPIHandler:
    def __init__(self):
        self.bearer_token = X_BEARER_TOKEN
        self.headers = {"Authorization": f"Bearer {self.bearer_token}"} if self.bearer_token else {}
        self.base_url = "https://api.twitter.com/2"

        if not self.bearer_token:
            raise ValueError("X_BEARER_TOKEN not found or empty in environment (.env).")

        print(f"XAPIHandler initialized (token length: {len(self.bearer_token)})")

    def get_tweet_id(self, url):
        print(f"[XAPI] get_tweet_id: url={url}")
        match = re.search(r'/status/(\d+)', url)
        tweet_id = match.group(1) if match else None
        print(f"[XAPI] extracted tweet_id={tweet_id}")
        return tweet_id

    def _choose_media_url(self, media):
        # Prefer direct image/video url fields when present
        for key in ("url", "preview_image_url", "media_url_https"):
            if media.get(key):
                return media.get(key)
        # videos may have 'variants' with 'url'
        variants = media.get("variants") or []
        for v in variants:
            if v.get("url"):
                return v.get("url")
        return None

    def fetch_post(self, tweet_id):
        print(f"[XAPI] fetch_post: id={tweet_id}")
        params = {
            "expansions": "author_id,attachments.media_keys",
            "tweet.fields": "created_at,public_metrics,attachments,entities",
            "user.fields": "username,verified",
            "media.fields": "url,preview_image_url,type,variants,media_key"
        }

        try:
            r = requests.get(f"{self.base_url}/tweets/{tweet_id}", headers=self.headers, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"[XAPI] RequestException: {e}")
            traceback.print_exc()
            raise

        print(f"[XAPI] status={r.status_code}")
        body = None
        try:
            body = r.json()
        except Exception:
            print("[XAPI] response is not JSON; raw text preview:")
            print(r.text[:1000])
            r.raise_for_status()

        if r.status_code == 401 or r.status_code == 403:
            print(f"[XAPI] Auth/Permission error ({r.status_code}): {body}")
            raise ValueError(f"Auth/Permission error ({r.status_code}): {body}")

        if r.status_code != 200:
            print(f"[XAPI] Non-200 response: {r.status_code} body={body}")
            raise ValueError(f"X API error {r.status_code}: {body}")

        # Validate structure
        if not isinstance(body, dict) or "data" not in body:
            print(f"[XAPI] No 'data' in response: {body}")
            raise ValueError(f"No 'data' in response: {body}")

        data = body["data"]
        includes = body.get("includes", {})

        # Author
        author = {}
        for u in includes.get("users", []):
            if u.get("id") == data.get("author_id"):
                author = u
                break

        # Media
        media_urls = []
        media_list = includes.get("media", [])
        media_keys = data.get("attachments", {}).get("media_keys", [])
        for mk in media_keys:
            media = next((m for m in media_list if m.get("media_key") == mk or m.get("media_key") == mk), None)
            if media:
                chosen = self._choose_media_url(media)
                if chosen:
                    media_urls.append(chosen)
                else:
                    print(f"[XAPI] media present but no usable url for media: {media}")

        result = {
            "text": data.get("text", ""),
            "author": author.get("username") if author else None,
            "verified": author.get("verified", False),
            "metrics": data.get("public_metrics", {}),
            "media_urls": media_urls
        }

        print(f"[XAPI] fetched result summary: author={result['author']}, text_len={len(result['text'])}, media_count={len(media_urls)}, metrics={result['metrics']}")
        return result

    def analyze_url(self, url):
        print(f"[XAPI] analyze_url: {url}")
        tweet_id = self.get_tweet_id(url)
        if not tweet_id:
            raise ValueError("Invalid X URL - couldn't extract tweet id.")
        return self.fetch_post(tweet_id)

