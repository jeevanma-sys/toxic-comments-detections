"""
src/reddit_producer.py

Fetches live Reddit comments for a given subreddit or post and publishes
each one as a JSON message to Kafka (or direct fallback).

Run:
  python src/reddit_producer.py --target "technology" --poll_interval 10
  python src/reddit_producer.py --target "https://www.reddit.com/r/technology/comments/..."
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reddit_source import RedditSource

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "youtube-comments"  # unified topic for live comments or 'reddit-comments'


def main(args):
    source = RedditSource(target=args.target)
    meta = source.fetch_meta()
    print(f"\n[RedditProducer] Connected to target: {meta.get('title')} ({meta.get('url')})")

    producer = None
    if not args.no_kafka:
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=args.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print(f"[RedditProducer] Connected to Kafka at {args.bootstrap_servers} (topic: {args.topic})")
        except Exception as e:
            print(f"[RedditProducer] Kafka unavailable ({e}). Running in standalone monitor mode.")
            producer = None

    print(f"[RedditProducer] Polling every {args.poll_interval}s ... (Ctrl+C to stop)")

    total_seen = 0
    try:
        while True:
            new_comments = source.poll_new_comments(limit=args.limit)
            if new_comments:
                for c in new_comments:
                    total_seen += 1
                    message = {
                        "comment_id": c["comment_id"],
                        "source": "reddit",
                        "video_id": c["source_id"],  # mapped to video_id for backward compatibility
                        "source_id": c["source_id"],
                        "author": c["author"],
                        "text": c["text"],
                        "published_at": c["published_at"],
                        "fetched_at": datetime.utcnow().isoformat(),
                        "url": c.get("url", ""),
                    }

                    if producer is not None:
                        producer.send(args.topic, value=message)

                    print(f"-> [{c['author']}]: {c['text'][:80]}...")

                if producer is not None:
                    producer.flush()

                print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] Ingested {len(new_comments)} new Reddit comments (Total: {total_seen})")

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\n[RedditProducer] Stopped by user.")
    finally:
        if producer is not None:
            producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="technology", help="Subreddit name or Reddit post URL")
    parser.add_argument("--limit", type=int, default=15, help="Batch limit per poll")
    parser.add_argument("--poll_interval", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument("--bootstrap_servers", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", type=str, default=KAFKA_TOPIC)
    parser.add_argument("--no_kafka", action="store_true", help="Print only, do not send to Kafka")
    args = parser.parse_args()
    main(args)
