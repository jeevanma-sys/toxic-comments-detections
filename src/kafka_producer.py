"""
kafka_producer.py

Fetches live YouTube comments for a given video and publishes each
one as a JSON message to the Kafka topic "youtube-comments".

This replaces the old direct-polling-and-scoring loop: now ingestion
(this script) is decoupled from scoring (done by the Kafka consumer
running inside web/app.py).

Run:
  python src/kafka_producer.py --video_id VIDEO_ID --api_key YOUR_KEY
"""

import argparse
import json
import time
from datetime import datetime

from kafka import KafkaProducer
from googleapiclient.discovery import build

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "youtube-comments"


def fetch_comments(youtube, video_id):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=50,
        order="time",
        textFormat="plainText",
    )
    return request.execute()


def main(args):
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS} ...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    youtube = build("youtube", "v3", developerKey=args.api_key)
    seen_comment_ids = set()

    print(f"Polling video {args.video_id} every {args.poll_interval}s, "
          f"publishing to topic '{KAFKA_TOPIC}' ... (Ctrl+C to stop)")

    try:
        while True:
            try:
                response = fetch_comments(youtube, args.video_id)
            except Exception as e:
                print(f"Fetch error (will retry): {e}")
                time.sleep(args.poll_interval)
                continue

            new_count = 0
            for item in response.get("items", []):
                comment_id = item["snippet"]["topLevelComment"]["id"]
                if comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)

                snippet = item["snippet"]["topLevelComment"]["snippet"]
                message = {
                    "comment_id": comment_id,
                    "video_id": args.video_id,
                    "author": snippet.get("authorDisplayName"),
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "fetched_at": datetime.utcnow().isoformat(),
                }

                producer.send(KAFKA_TOPIC, value=message)
                new_count += 1
                print(f"-> sent: {message['text'][:70]}")

            producer.flush()
            print(f"[{datetime.utcnow().isoformat()}] Published {new_count} new comments "
                  f"(total seen: {len(seen_comment_ids)})")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--poll_interval", type=int, default=15)
    args = parser.parse_args()
    main(args)