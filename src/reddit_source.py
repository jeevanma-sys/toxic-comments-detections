"""
src/reddit_source.py

Live Reddit Ingestion Source:
  - Supports live comment ingestion from:
      1. Subreddits (e.g. "r/technology", "r/AskReddit", "r/worldnews", "r/all")
      2. Specific Reddit threads/posts (e.g. "https://www.reddit.com/r/.../comments/...")
  - Dual Mode Support:
      - Authenticated Reddit OAuth API (when client_id / secret are provided)
      - Zero-setup Browser Feed mode (fetches public comments without requiring API keys!)
  - Standardizes comments for the toxicity scoring pipeline.
"""

import os
import re
import html
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "ToxicCommentMonitor/1.0")


def clean_html(raw_html: str) -> str:
    """Strip HTML tags and unescape entities from Reddit comments."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ").strip()
    return html.unescape(text)


def parse_reddit_target(target: str) -> dict:
    """
    Parse a Reddit URL or shorthand string:
      - "technology" -> subreddit "technology"
      - "r/AskReddit" -> subreddit "AskReddit"
      - "https://reddit.com/r/technology/comments/123abc/title/" -> post_id "123abc", sub "technology"
    """
    target = target.strip()

    # Match post URL
    post_match = re.search(r"reddit\.com/r/([^/]+)/comments/([a-zA-Z0-9]+)", target)
    if post_match:
        return {
            "type": "post",
            "subreddit": post_match.group(1),
            "post_id": post_match.group(2),
            "target": target,
        }

    # Match short comment URL: reddit.com/comments/123abc
    short_match = re.search(r"reddit\.com/comments/([a-zA-Z0-9]+)", target)
    if short_match:
        return {
            "type": "post",
            "subreddit": None,
            "post_id": short_match.group(1),
            "target": target,
        }

    # Match subreddit shorthand: "r/technology" or "technology"
    sub_clean = re.sub(r"^/?r/", "", target).strip().strip("/")
    if "/" not in sub_clean and sub_clean:
        return {
            "type": "subreddit",
            "subreddit": sub_clean,
            "post_id": None,
            "target": sub_clean,
        }

    return {
        "type": "subreddit",
        "subreddit": sub_clean or "all",
        "post_id": None,
        "target": target,
    }


class RedditSource:
    """
    Ingests live comments from Reddit with zero-setup RSS fallback
    or official OAuth client if credentials are configured.
    """

    def __init__(
        self,
        target: str = "technology",
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = None,
    ):
        self.parsed = parse_reddit_target(target)
        self.client_id = client_id or REDDIT_CLIENT_ID
        self.client_secret = client_secret or REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or REDDIT_USER_AGENT
        self.seen_comment_ids = set()
        self.token = None
        self.token_expiry = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def _authenticate_oauth(self) -> bool:
        """Authenticate with Reddit API using OAuth client_credentials grant."""
        if not (self.client_id and self.client_secret):
            return False

        if self.token and time.time() < self.token_expiry - 60:
            return True

        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {"grant_type": "client_credentials"}
            headers = {"User-Agent": self.user_agent}
            resp = requests.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                payload = resp.json()
                self.token = payload["access_token"]
                self.token_expiry = time.time() + payload.get("expires_in", 3600)
                return True
        except Exception as e:
            print(f"[RedditSource] OAuth authentication failed: {e}")
        return False

    def fetch_meta(self) -> dict:
        """Fetch metadata about the subreddit or target post."""
        sub = self.parsed.get("subreddit") or "all"
        post_id = self.parsed.get("post_id")

        meta = {
            "source": "reddit",
            "type": self.parsed["type"],
            "subreddit": f"r/{sub}",
            "title": f"Live stream from r/{sub}",
            "author": "Reddit Community",
            "url": f"https://www.reddit.com/r/{sub}/",
            "subscriber_count": "Active",
        }

        if post_id:
            try:
                feed_url = f"https://www.reddit.com/comments/{post_id}.rss"
                resp = self.session.get(feed_url, timeout=10)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    title_elem = root.find("atom:title", ns)
                    if title_elem is not None and title_elem.text:
                        meta["title"] = title_elem.text
                    meta["url"] = f"https://www.reddit.com/comments/{post_id}"
            except Exception:
                pass

        return meta

    def fetch_comments(self, limit: int = 25) -> list[dict]:
        """
        Fetch the newest comments. Uses OAuth if available,
        otherwise fetches via the robust live feed endpoint.
        """
        if self._authenticate_oauth():
            return self._fetch_oauth(limit=limit)
        return self._fetch_feed(limit=limit)

    def _fetch_feed(self, limit: int = 25) -> list[dict]:
        """Fetch latest comments using Reddit live RSS/Atom feed."""
        sub = self.parsed.get("subreddit") or "all"
        post_id = self.parsed.get("post_id")

        if post_id:
            feed_url = f"https://www.reddit.com/comments/{post_id}.rss?limit={limit}"
        else:
            feed_url = f"https://www.reddit.com/r/{sub}/comments.rss?limit={limit}"

        comments = []
        try:
            resp = self.session.get(feed_url, timeout=12)
            if resp.status_code != 200:
                print(f"[RedditSource] Feed returned status {resp.status_code}")
                return []

            root = ET.fromstring(resp.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall("atom:entry", ns):
                entry_id = entry.findtext("atom:id", default="", namespaces=ns)
                comment_id = entry_id.split("/")[-1] if entry_id else f"rc_{int(time.time()*1000)}"

                author_elem = entry.find("atom:author/atom:name", ns)
                author = author_elem.text if author_elem is not None else "anonymous"
                author = re.sub(r"^/u/", "", author)

                content_elem = entry.find("atom:content", ns)
                raw_html = content_elem.text if content_elem is not None else ""
                body = clean_html(raw_html)

                # Skip blank comments or author deleted
                if not body or body == "[deleted]" or body == "[removed]":
                    continue

                updated = entry.findtext("atom:updated", default=datetime.utcnow().isoformat(), namespaces=ns)
                link_elem = entry.find("atom:link", ns)
                permalink = link_elem.get("href") if link_elem is not None else f"https://reddit.com/r/{sub}"

                comments.append({
                    "comment_id": comment_id,
                    "source": "reddit",
                    "source_id": f"r/{sub}" if not post_id else post_id,
                    "author": f"u/{author}",
                    "text": body,
                    "published_at": updated,
                    "url": permalink,
                })

        except Exception as e:
            print(f"[RedditSource] Error parsing Reddit feed: {e}")

        return comments

    def _fetch_oauth(self, limit: int = 25) -> list[dict]:
        """Fetch comments using Reddit OAuth REST API."""
        sub = self.parsed.get("subreddit") or "all"
        post_id = self.parsed.get("post_id")
        headers = {
            "Authorization": f"bearer {self.token}",
            "User-Agent": self.user_agent,
        }

        if post_id:
            api_url = f"https://oauth.reddit.com/comments/{post_id}?limit={limit}"
        else:
            api_url = f"https://oauth.reddit.com/r/{sub}/comments?limit={limit}"

        comments = []
        try:
            resp = requests.get(api_url, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                items = []
                if isinstance(data, list) and len(data) > 1:
                    items = data[1].get("data", {}).get("children", [])
                elif isinstance(data, dict):
                    items = data.get("data", {}).get("children", [])

                for item in items:
                    cdata = item.get("data", {})
                    cid = cdata.get("id")
                    body = cdata.get("body", "").strip()
                    if not body or body in ["[deleted]", "[removed]"]:
                        continue

                    comments.append({
                        "comment_id": cid or f"rc_{int(time.time()*1000)}",
                        "source": "reddit",
                        "source_id": f"r/{sub}" if not post_id else post_id,
                        "author": f"u/{cdata.get('author', 'anonymous')}",
                        "text": body,
                        "published_at": datetime.fromtimestamp(cdata.get("created_utc", time.time())).isoformat(),
                        "url": f"https://reddit.com{cdata.get('permalink', '')}",
                    })
        except Exception as e:
            print(f"[RedditSource] OAuth fetch error: {e}")

        return comments

    def poll_new_comments(self, limit: int = 25) -> list[dict]:
        """Fetch only new comments that haven't been seen yet."""
        all_comments = self.fetch_comments(limit=limit)
        new_comments = []
        for c in all_comments:
            cid = c["comment_id"]
            if cid not in self.seen_comment_ids:
                self.seen_comment_ids.add(cid)
                new_comments.append(c)
        return new_comments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="technology", help="Subreddit or post URL")
    parser.add_argument("--limit", type=int, default=5, help="Number of comments to fetch")
    args = parser.parse_args()

    print(f"Connecting to Reddit target: {args.target} ...")
    source = RedditSource(target=args.target)
    meta = source.fetch_meta()
    print("Source Meta:", meta)
    comments = source.fetch_comments(limit=args.limit)
    print(f"Fetched {len(comments)} comments:")
    for i, c in enumerate(comments[:args.limit], 1):
        print(f"\n[{i}] {c['author']} ({c['published_at']}):")
        print(f"    {c['text'][:120]}...")
