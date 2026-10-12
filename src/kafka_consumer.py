"""
src/kafka_consumer.py

Consume live comments (YouTube or Reddit) from Kafka, score them using
Transformer Context-Aware Scorer (or TF-IDF), and append results to CSV.

Run:
  python src/kafka_consumer.py --model transformer
  python src/kafka_consumer.py --model mitigated
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
import joblib
from kafka import KafkaConsumer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformer_scorer import (
    get_transformer_scorer,
    clean_text_bow,
    clean_text_simple,
    detect_identities,
    MITIGATED_MODEL_PATH,
    MITIGATED_VEC_PATH,
)

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "youtube-comments"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_csv(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "comment_id", "video_id", "author", "text", "clean_text",
                "toxicity_prob", "is_toxic", "identity_flags", "scored_at",
            ])


def consume(args):
    ensure_csv(args.output_csv)

    print(f"[KafkaConsumer] Loading model: {args.model} ...")
    if args.model == "transformer":
        transformer_scorer = get_transformer_scorer()
        model, vectorizer = None, None
    else:
        transformer_scorer = None
        model = joblib.load(args.model_path)
        vectorizer = joblib.load(args.vectorizer_path)

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
        auto_offset_reset=args.offset_reset,
        enable_auto_commit=True,
        group_id=args.group_id,
    )
    print(f"[KafkaConsumer] Consuming '{args.topic}' from {args.bootstrap_servers} ... (Ctrl+C to stop)")

    try:
        for message in consumer:
            data = message.value
            source_id = data.get("video_id") or data.get("source_id")
            if args.video_id and source_id != args.video_id:
                continue

            raw_text = data.get("text", "")
            cleaned = clean_text_simple(raw_text)

            if args.model == "transformer":
                score_info = transformer_scorer.score_text(raw_text)
                probability = score_info["toxicity_prob"]
                is_toxic = score_info["is_toxic"]
            else:
                bow_clean = clean_text_bow(raw_text)
                probability = float(model.predict_proba(vectorizer.transform([bow_clean]))[0, 1])
                is_toxic = probability >= args.threshold

            identity_flags = detect_identities(raw_text)
            now_iso = datetime.now(timezone.utc).isoformat()

            with open(args.output_csv, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    data.get("comment_id"), source_id, data.get("author"), raw_text,
                    cleaned, round(probability, 4), is_toxic,
                    ",".join(identity_flags), now_iso,
                ])

            source_label = data.get("source", "stream").upper()
            status_label = "TOXIC" if is_toxic else "CLEAN"
            print(f"[{source_label} | {status_label} {probability:.3f}] {data.get('author')}: {raw_text[:70]}")

    except KeyboardInterrupt:
        print("\n[KafkaConsumer] Stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap_servers", type=str, default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", type=str, default=KAFKA_TOPIC)
    parser.add_argument("--group_id", type=str, default="toxicity-scorer-group")
    parser.add_argument("--offset_reset", type=str, default="latest")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "mitigated"])
    parser.add_argument("--model_path", type=str, default=MITIGATED_MODEL_PATH)
    parser.add_argument("--vectorizer_path", type=str, default=MITIGATED_VEC_PATH)
    parser.add_argument("--output_csv", type=str, default=os.path.join(BASE_DIR, "dashboard", "live_predictions.csv"))
    parser.add_argument("--video_id", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    consume(args)