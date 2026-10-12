"""
web/app.py

Flask backend with:
  1. Transformer-based context-aware scoring (Toxic-BERT) with TF-IDF fallback.
  2. LIME Explainability API (/api/explain) for token-level attribution.
  3. Bias Dashboard API (/api/bias_metrics) for fairness metrics (BPSN/BNSP AUC, FPR parity).
  4. Dual live streaming sources:
       - YouTube comments (via YouTube Data API)
       - Reddit live comments (via Reddit live feed / OAuth API)
  5. Resilient direct scoring with optional Kafka fan-out.
"""

import os
import re
import csv
import json
import time
import threading
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, jsonify, render_template, request

# Import our custom modules
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "src"))

from transformer_scorer import (
    get_transformer_scorer,
    clean_text_simple,
    clean_text_bow,
    detect_identities,
    MITIGATED_MODEL_PATH,
    MITIGATED_VEC_PATH,
)
from explainability import get_explainer
from reddit_source import RedditSource, parse_reddit_target

CSV_PATH = os.path.join(PROJECT_DIR, "dashboard", "live_predictions.csv")
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "youtube-comments"
DEFAULT_YT_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyC18uJp6Vxa7zSBGHL72m9FPUaFQ7aQqNM")

app = Flask(__name__)

# Pre-warm scorer and explainer
print("[Flask] Pre-warming Transformer Scorer and LIME Explainer ...")
transformer_scorer = get_transformer_scorer()
explainer = get_explainer()

# Global live state
state = {
    "source": "youtube",         # "youtube" | "reddit"
    "target": None,              # video_id or subreddit / reddit url
    "target_title": None,
    "api_key": DEFAULT_YT_API_KEY,
    "model_type": "transformer", # "transformer" | "mitigated"
    "status": "idle",            # "idle" | "streaming" | "error"
    "last_error": None,
    "has_kafka": False,
}
state_lock = threading.Lock()

# Attempt Kafka producer connection
kafka_producer = None
try:
    from kafka import KafkaProducer
    kafka_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        request_timeout_ms=3000,
    )
    with state_lock:
        state["has_kafka"] = True
    print("[Flask] Kafka Producer connected.")
except Exception as e:
    print(f"[Flask] Kafka not active ({e}). Using direct resilient streaming.")


# Global in-memory set to prevent duplicate writes to CSV
seen_comment_keys = set()


def ensure_csv():
    if not os.path.exists(CSV_PATH):
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "comment_id", "video_id", "author", "text", "clean_text",
                "toxicity_prob", "is_toxic", "identity_flags", "scored_at",
            ])


def score_and_record(comment_data, model_choice="transformer"):
    """Scores a single incoming comment and appends to live_predictions.csv (9 standard columns)."""
    ensure_csv()
    comment_id = str(comment_data.get("comment_id", f"c_{int(time.time()*1000)}"))
    target_id = str(comment_data.get("video_id") or comment_data.get("source_id", "unknown"))

    dedup_key = f"{target_id}::{comment_id}"
    if dedup_key in seen_comment_keys:
        return 0.0, False
    seen_comment_keys.add(dedup_key)

    raw_text = comment_data.get("text", "")
    author = comment_data.get("author", "Anonymous")

    if model_choice == "transformer":
        score_info = transformer_scorer.score_text(raw_text)
        toxicity_prob = score_info["toxicity_prob"]
        is_toxic = score_info["is_toxic"]
    else:
        # BoW Mitigated
        clf, vec = explainer._get_mitigated_model()
        if clf and vec:
            cleaned_bow = clean_text_bow(raw_text)
            toxicity_prob = float(clf.predict_proba(vec.transform([cleaned_bow]))[0, 1])
            is_toxic = toxicity_prob >= 0.5
        else:
            score_info = transformer_scorer.score_text(raw_text)
            toxicity_prob = score_info["toxicity_prob"]
            is_toxic = score_info["is_toxic"]

    identities = detect_identities(raw_text)
    now_iso = datetime.now(timezone.utc).isoformat()

    row = [
        comment_id, target_id, author, raw_text, clean_text_simple(raw_text),
        round(toxicity_prob, 4), is_toxic, ",".join(identities), now_iso,
    ]

    with open(CSV_PATH, "a", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return toxicity_prob, is_toxic


# ---------------- PRODUCER: YouTube ----------------
def produce_youtube_loop(video_id, api_key, model_choice):
    from googleapiclient.discovery import build
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        with state_lock:
            state["status"] = "error"
            state["last_error"] = f"YouTube API init failed: {e}"
        return

    seen_ids = set()
    first_poll = True

    while True:
        with state_lock:
            if state["target"] != video_id or state["source"] != "youtube":
                break

        try:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=50,
                order="time", textFormat="plainText",
            ).execute()

            scored_in_batch = 0
            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]
                comment_id = top_comment["id"]
                if comment_id in seen_ids:
                    continue
                seen_ids.add(comment_id)

                snippet = top_comment["snippet"]
                msg = {
                    "comment_id": comment_id,
                    "video_id": video_id,
                    "author": snippet.get("authorDisplayName"),
                    "text": snippet.get("textDisplay", ""),
                    "published_at": snippet.get("publishedAt"),
                    "source": "youtube",
                }

                # Always score directly so UI updates with immediate feedback
                score_and_record(msg, model_choice=model_choice)
                scored_in_batch += 1

                # Optional Kafka fan-out
                if kafka_producer is not None:
                    try:
                        kafka_producer.send(KAFKA_TOPIC, value=msg)
                    except Exception:
                        pass

            if kafka_producer is not None:
                try:
                    kafka_producer.flush(timeout=2)
                except Exception:
                    pass

            with state_lock:
                state["status"] = "streaming"
                state["last_error"] = None

            first_poll = False

        except Exception as e:
            err_msg = str(e)
            if "commentsDisabled" in err_msg:
                err_msg = "Comments are disabled on this video."
            elif "quotaExceeded" in err_msg:
                err_msg = "YouTube API daily quota exceeded."
            elif "videoNotFound" in err_msg:
                err_msg = "Video not found. Please check the video link."
            print(f"[YouTube Producer] Error for {video_id}: {err_msg}")
            with state_lock:
                state["status"] = "error"
                state["last_error"] = err_msg

        time.sleep(10)


# ---------------- PRODUCER: Reddit ----------------
def produce_reddit_loop(target_str, model_choice):
    source = RedditSource(target=target_str)
    seen_ids = set()

    meta = source.fetch_meta()
    with state_lock:
        state["target_title"] = meta.get("title")

    while True:
        with state_lock:
            if state["target"] != target_str or state["source"] != "reddit":
                break

        try:
            comments = source.fetch_comments(limit=25)
            for c in comments:
                cid = c["comment_id"]
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)

                msg = {
                    "comment_id": cid,
                    "video_id": c["source_id"],
                    "source_id": c["source_id"],
                    "author": c["author"],
                    "text": c["text"],
                    "published_at": c["published_at"],
                    "url": c.get("url", ""),
                    "source": "reddit",
                }

                # Always score directly for immediate responsiveness
                score_and_record(msg, model_choice=model_choice)

                if kafka_producer is not None:
                    try:
                        kafka_producer.send(KAFKA_TOPIC, value=msg)
                    except Exception:
                        pass

            if kafka_producer is not None:
                try:
                    kafka_producer.flush(timeout=2)
                except Exception:
                    pass

            with state_lock:
                state["status"] = "streaming"
                state["last_error"] = None

        except Exception as e:
            print(f"[Reddit Producer] Error for {target_str}: {e}")
            with state_lock:
                state["status"] = "error"
                state["last_error"] = str(e)

        time.sleep(10)


# ---------------- CONSUMER: Kafka loop ----------------
def kafka_consumer_loop():
    """Optional Kafka consumer thread."""
    ensure_csv()
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="toxicity-web-group",
            consumer_timeout_ms=5000,
        )
        print(f"[Kafka Consumer] Listening on topic '{KAFKA_TOPIC}'.")
    except Exception as e:
        print(f"[Kafka Consumer] Not starting (Kafka unavailable: {e}).")
        return

    while True:
        try:
            for message in consumer:
                with state_lock:
                    model_choice = state["model_type"]
                data = message.value
                score_and_record(data, model_choice=model_choice)
        except Exception:
            time.sleep(2)


def extract_youtube_id(url_or_id):
    url_or_id = url_or_id.strip()
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return None


# ---------------- HTTP ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)
    source_type = data.get("source", "youtube").lower()
    input_val = data.get("url", "").strip()
    model_choice = data.get("model", "transformer").lower()

    if not input_val:
        return jsonify({"ok": False, "error": "Please provide a valid URL or target."}), 400

    with state_lock:
        state["source"] = source_type
        state["model_type"] = model_choice
        state["status"] = "streaming"
        state["last_error"] = None

    if source_type == "reddit":
        parsed = parse_reddit_target(input_val)
        sub = parsed.get("subreddit") or "all"
        target_str = parsed.get("post_id") or f"r/{sub}"
        with state_lock:
            state["target"] = target_str

        thread = threading.Thread(
            target=produce_reddit_loop,
            args=(target_str, model_choice),
            daemon=True,
        )
        thread.start()
        return jsonify({"ok": True, "source": "reddit", "target": target_str})

    else:
        # YouTube
        video_id = extract_youtube_id(input_val)
        if not video_id:
            return jsonify({"ok": False, "error": "Could not extract a valid YouTube video ID from that link."}), 400

        with state_lock:
            state["target"] = video_id
            state["api_key"] = DEFAULT_YT_API_KEY

        thread = threading.Thread(
            target=produce_youtube_loop,
            args=(video_id, DEFAULT_YT_API_KEY, model_choice),
            daemon=True,
        )
        thread.start()
        return jsonify({"ok": True, "source": "youtube", "target": video_id})


@app.route("/api/data")
def api_data():
    with state_lock:
        active_target = state["target"]
        source_type = state["source"]
        status = state["status"]
        last_error = state["last_error"]
        model_choice = state["model_type"]

    if not active_target or not os.path.exists(CSV_PATH):
        return jsonify({
            "total": 0, "toxicity_rate": 0, "identity_count": 0,
            "target": active_target, "source": source_type, "model": model_choice,
            "comments": [], "identity_rates": [],
            "status": status, "last_error": last_error,
        })

    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
        df["identity_flags"] = df["identity_flags"].fillna("")
        df = df[df["video_id"].astype(str) == str(active_target)]
    except Exception as e:
        return jsonify({"error": str(e), "comments": []})

    filter_mode = request.args.get("filter", "all")
    total = len(df)
    toxicity_rate = round(float(df["is_toxic"].mean()) * 100, 1) if total else 0
    identity_count = int((df["identity_flags"] != "").sum())

    view_df = df.sort_values("scored_at", ascending=False)
    if filter_mode == "toxic":
        view_df = view_df[view_df["is_toxic"] == True]
    elif filter_mode == "clean":
        view_df = view_df[view_df["is_toxic"] == False]

    comments = []
    for _, row in view_df.head(50).iterrows():
        comments.append({
            "comment_id": str(row.get("comment_id", "")),
            "author": row.get("author", "Unknown"),
            "text": row.get("text", ""),
            "toxicity_prob": float(row.get("toxicity_prob", 0.0)),
            "is_toxic": bool(row.get("is_toxic", False)),
            "identity_flags": str(row.get("identity_flags", "")),
            "scored_at": str(row.get("scored_at", "")),
            "source": source_type,
        })

    exploded = df.assign(identity_flags=df["identity_flags"].str.split(",")).explode("identity_flags")
    exploded = exploded[exploded["identity_flags"].fillna("") != ""]
    identity_rates = []
    if not exploded.empty:
        grouped = exploded.groupby("identity_flags")["is_toxic"].mean().sort_values(ascending=False)
        identity_rates = [{"identity": k, "rate": round(float(v) * 100, 1)} for k, v in grouped.items()]

    return jsonify({
        "total": total,
        "toxicity_rate": toxicity_rate,
        "identity_count": identity_count,
        "target": active_target,
        "source": source_type,
        "model": model_choice,
        "comments": comments,
        "identity_rates": identity_rates,
        "status": status,
        "last_error": last_error,
    })


@app.route("/api/source_meta")
def api_source_meta():
    with state_lock:
        active_target = state["target"]
        source_type = state["source"]
        api_key = state["api_key"]
        target_title = state["target_title"]

    if not active_target:
        return jsonify({})

    if source_type == "reddit":
        src = RedditSource(target=active_target)
        meta = src.fetch_meta()
        if target_title:
            meta["title"] = target_title
        return jsonify(meta)

    # YouTube
    if not api_key:
        return jsonify({})
    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(part="snippet,statistics", id=active_target).execute()
        items = resp.get("items", [])
        if not items:
            return jsonify({})
        snippet = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        return jsonify({
            "source": "youtube",
            "title": snippet.get("title"),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle"),
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
        })
    except Exception:
        return jsonify({})


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """LIME Explainability Endpoint."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    model_choice = data.get("model", "transformer").lower()

    if not text:
        return jsonify({"error": "No text provided for explanation"}), 400

    try:
        result = explainer.explain_comment(
            text=text,
            model_type=model_choice,
            num_features=8,
            num_samples=100,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/bias_metrics")
def api_bias_metrics():
    """Bias & Fairness Dashboard Endpoint."""
    comp_file = os.path.join(PROJECT_DIR, "models", "before_after_comparison.csv")
    fpr_file = os.path.join(PROJECT_DIR, "models", "fpr_before_after_comparison.csv")

    response_data = {
        "identities": [],
        "overall": {},
    }

    if os.path.exists(comp_file) and os.path.exists(fpr_file):
        df_comp = pd.read_csv(comp_file)
        df_fpr = pd.read_csv(fpr_file)
        merged = df_comp.merge(df_fpr, on="identity")

        identities_list = []
        for _, row in merged.iterrows():
            identities_list.append({
                "identity": row["identity"],
                "subgroup_auc_baseline": round(float(row.get("subgroup_auc_baseline", 0.0)), 4),
                "subgroup_auc_mitigated": round(float(row.get("subgroup_auc_mitigated", 0.0)), 4),
                "bpsn_auc_baseline": round(float(row.get("bpsn_auc_baseline", 0.0)), 4),
                "bpsn_auc_mitigated": round(float(row.get("bpsn_auc_mitigated", 0.0)), 4),
                "fpr_baseline": round(float(row.get("fpr_baseline", 0.0)) * 100, 2),
                "fpr_mitigated": round(float(row.get("fpr_mitigated", 0.0)) * 100, 2),
                "fpr_postproc": round(float(row.get("fpr_postproc", 0.0)) * 100, 2),
                "fnr_baseline": round(float(row.get("fnr_baseline", 0.0)) * 100, 2),
                "fnr_mitigated": round(float(row.get("fnr_mitigated", 0.0)) * 100, 2),
            })

        bpsn_base_mean = df_comp["bpsn_auc_baseline"].mean()
        bpsn_mit_mean = df_comp["bpsn_auc_mitigated"].mean()
        bpsn_gain = (bpsn_mit_mean - bpsn_base_mean) / bpsn_base_mean * 100

        max_fpr_base = df_fpr["fpr_baseline"].max() * 100
        min_fpr_base = df_fpr["fpr_baseline"].min() * 100
        fpr_disparity_base = max_fpr_base - min_fpr_base

        max_fpr_post = df_fpr["fpr_postproc"].max() * 100
        min_fpr_post = df_fpr["fpr_postproc"].min() * 100
        fpr_disparity_post = max_fpr_post - min_fpr_post

        response_data["identities"] = identities_list
        response_data["overall"] = {
            "mean_bpsn_baseline": round(float(bpsn_base_mean), 4),
            "mean_bpsn_mitigated": round(float(bpsn_mit_mean), 4),
            "bpsn_gain_pct": round(float(bpsn_gain), 2),
            "fpr_disparity_baseline": round(float(fpr_disparity_base), 1),
            "fpr_disparity_postproc": round(float(fpr_disparity_post), 1),
            "disparity_reduction_pct": round(float((fpr_disparity_base - fpr_disparity_post) / fpr_disparity_base * 100), 1),
        }

    return jsonify(response_data)


@app.route("/api/model_info")
def api_model_info():
    return jsonify({
        "transformer_model": "unitary/toxic-bert",
        "transformer_backend": transformer_scorer.backend,
        "supports_lime": True,
        "supports_sources": ["youtube", "reddit"],
        "kafka_connected": state["has_kafka"],
    })


if __name__ == "__main__":
    if state["has_kafka"]:
        consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
        consumer_thread.start()
    app.run(debug=False, port=5000)