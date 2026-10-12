"""
src/realtime_scorer.py

Real-time pipeline with support for:
  1. Transformer Context-Aware Scorer (Toxic-BERT) OR Mitigated/Baseline BoW models
  2. Multiple Live Sources: YouTube Data API OR Live Reddit Stream (r/sub or post)
  3. Identity detection & real-time fairness tagging
  4. Live CSV output for dashboard consumption

Run with YouTube:
  python src/realtime_scorer.py --source youtube --video_id VIDEO_ID --api_key YOUR_KEY --model transformer

Run with Reddit:
  python src/realtime_scorer.py --source reddit --target "technology" --model transformer
"""

import argparse
import re
import time
import os
import csv
import joblib
from datetime import datetime, timezone

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformer_scorer import get_transformer_scorer, clean_text_bow, clean_text_simple, detect_identities
from reddit_source import RedditSource


def ensure_csv(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "comment_id", "video_id", "author", "text", "clean_text",
                "toxicity_prob", "is_toxic", "identity_flags", "scored_at",
            ])


def fetch_youtube_comments(youtube, video_id, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="time",
        textFormat="plainText",
    )
    return request.execute()


def main(args):
    ensure_csv(args.output_csv)

    # 1. Initialize scoring model
    scorer_type = args.model.lower()
    print(f"[RealtimeScorer] Loading scoring engine: {scorer_type} ...")

    transformer_scorer = None
    bow_model = None
    bow_vec = None

    if scorer_type == "transformer":
        transformer_scorer = get_transformer_scorer()
    else:
        if not os.path.exists(args.model_path) or not os.path.exists(args.vectorizer_path):
            print(f"[RealtimeScorer] BoW model not found at {args.model_path}. Falling back to Transformer.")
            transformer_scorer = get_transformer_scorer()
            scorer_type = "transformer"
        else:
            bow_model = joblib.load(args.model_path)
            bow_vec = joblib.load(args.vectorizer_path)

    # 2. Setup ingestion source
    seen_comment_ids = set()
    source_type = args.source.lower()

    if source_type == "reddit":
        target = args.target or args.video_id or "technology"
        print(f"[RealtimeScorer] Ingesting from Reddit: {target} (poll every {args.poll_interval}s)")
        reddit_source = RedditSource(target=target)
        meta = reddit_source.fetch_meta()
        print(f"[RealtimeScorer] Target: {meta.get('title')} ({meta.get('url')})")

        try:
            while True:
                comments = reddit_source.fetch_comments(limit=25)
                new_count = 0
                for c in comments:
                    cid = c["comment_id"]
                    if cid in seen_comment_ids:
                        continue
                    seen_comment_ids.add(cid)

                    raw_text = c["text"]
                    cleaned = clean_text_simple(raw_text)

                    if scorer_type == "transformer":
                        score_info = transformer_scorer.score_text(raw_text)
                        prob = score_info["toxicity_prob"]
                        is_toxic = score_info["is_toxic"]
                    else:
                        bow_clean = clean_text_bow(raw_text)
                        prob = float(bow_model.predict_proba(bow_vec.transform([bow_clean]))[0, 1])
                        is_toxic = prob >= args.threshold

                    identities = detect_identities(raw_text)
                    now_iso = datetime.now(timezone.utc).isoformat()

                    row = [
                        cid, c["source_id"], c["author"],
                        raw_text, cleaned, round(prob, 4), is_toxic,
                        ",".join(identities), now_iso,
                    ]
                    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)

                    new_count += 1
                    status_lbl = "TOXIC" if is_toxic else "CLEAN"
                    tag = f" [identities: {', '.join(identities)}]" if identities else ""
                    print(f"[{status_lbl} {prob:.3f}]{tag} {c['author']}: {raw_text[:70]}...")

                if new_count > 0:
                    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Scored {new_count} new Reddit comments.")

                time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            print("\n[RealtimeScorer] Stopped by user.")

    else:
        # YouTube ingestion
        if not args.api_key:
            print("Error: --api_key is required when using --source youtube")
            return
        if not args.video_id:
            print("Error: --video_id is required when using --source youtube")
            return

        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=args.api_key)
        print(f"[RealtimeScorer] Polling YouTube video {args.video_id} every {args.poll_interval}s ...")

        try:
            while True:
                response = fetch_youtube_comments(youtube, args.video_id)
                new_count = 0

                for item in response.get("items", []):
                    cid = item["snippet"]["topLevelComment"]["id"]
                    if cid in seen_comment_ids:
                        continue
                    seen_comment_ids.add(cid)

                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    raw_text = snippet.get("textDisplay", "")
                    cleaned = clean_text_simple(raw_text)

                    if scorer_type == "transformer":
                        score_info = transformer_scorer.score_text(raw_text)
                        prob = score_info["toxicity_prob"]
                        is_toxic = score_info["is_toxic"]
                    else:
                        bow_clean = clean_text_bow(raw_text)
                        prob = float(bow_model.predict_proba(bow_vec.transform([bow_clean]))[0, 1])
                        is_toxic = prob >= args.threshold

                    identities = detect_identities(raw_text)
                    now_iso = datetime.now(timezone.utc).isoformat()

                    row = [
                        cid, args.video_id, snippet.get("authorDisplayName"),
                        raw_text, cleaned, round(prob, 4), is_toxic,
                        ",".join(identities), now_iso,
                    ]
                    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)

                    new_count += 1
                    status_lbl = "TOXIC" if is_toxic else "CLEAN"
                    tag = f" [identities: {', '.join(identities)}]" if identities else ""
                    print(f"[{status_lbl} {prob:.3f}]{tag} {raw_text[:70]}...")

                if new_count > 0:
                    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Scored {new_count} new YouTube comments.")

                time.sleep(args.poll_interval)

        except KeyboardInterrupt:
            print("\n[RealtimeScorer] Stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="youtube", choices=["youtube", "reddit"],
                        help="Ingestion source: youtube or reddit")
    parser.add_argument("--target", type=str, default=None,
                        help="Reddit subreddit or post URL (for Reddit source)")
    parser.add_argument("--video_id", type=str, default=None,
                        help="YouTube video ID (for YouTube source)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="YouTube Data API key")
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "mitigated", "baseline"],
                        help="Model architecture: transformer (context-aware) or mitigated (TF-IDF)")
    parser.add_argument("--model_path", type=str, default="models/toxicity_model_mitigated.pkl")
    parser.add_argument("--vectorizer_path", type=str, default="models/tfidf_vectorizer_mitigated.pkl")
    parser.add_argument("--output_csv", type=str, default="dashboard/live_predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--poll_interval", type=int, default=12)
    args = parser.parse_args()
    main(args)