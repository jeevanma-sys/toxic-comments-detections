"""
test_e2e_features.py
Comprehensive End-to-End Test Suite verifying:
  1. Transformer Context-Aware Scorer
  2. LIME Explainability Engine & Token Highlights
  3. Bias & Fairness Metrics Extraction
  4. Reddit Live Ingestion Source
  5. Flask Web API Endpoints
"""

import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
sys.path.append(BASE_DIR)

from transformer_scorer import get_transformer_scorer
from explainability import get_explainer
from reddit_source import RedditSource, parse_reddit_target

def test_transformer_scorer():
    print("\n--- 1. Testing Transformer Context-Aware Scorer ---")
    scorer = get_transformer_scorer()
    assert scorer.backend in ["transformer", "tfidf_mitigated"], f"Unexpected backend: {scorer.backend}"
    
    # Test toxic
    res_toxic = scorer.score_text("You are an absolute idiot and a total failure")
    print(f"Toxic test prob: {res_toxic['toxicity_prob']:.4f}, is_toxic: {res_toxic['is_toxic']}")
    assert res_toxic["is_toxic"] == True, "Failed to identify toxic text"
    
    # Test benign identity expression
    res_identity = scorer.score_text("I am a proud gay woman and I love my community")
    print(f"Identity test prob: {res_identity['toxicity_prob']:.4f}, is_toxic: {res_identity['is_toxic']}")
    assert res_identity["is_toxic"] == False, "Falsely flagged benign identity expression"
    
    # Test predict_proba
    probs = scorer.predict_proba(["Clean comment", "You are stupid"])
    assert probs.shape == (2, 2), f"Shape mismatch: {probs.shape}"
    assert abs(probs[0].sum() - 1.0) < 1e-4, "Probabilities do not sum to 1"
    print("Transformer Scorer: PASSED [OK]")

def test_lime_explainability():
    print("\n--- 2. Testing LIME Explainability Module ---")
    explainer = get_explainer()
    test_text = "You are an idiot and completely useless"
    
    res = explainer.explain_comment(test_text, model_type="transformer", num_features=4, num_samples=60)
    assert "prediction" in res
    assert "features" in res
    assert len(res["features"]) > 0
    assert "highlighted_html" in res
    print(f"LIME prediction: {res['prediction']}")
    print(f"LIME features extracted: {[f['word'] for f in res['features']]}")
    print("LIME Explainability: PASSED [OK]")

def test_reddit_source():
    print("\n--- 3. Testing Reddit Live Source ---")
    parsed1 = parse_reddit_target("r/technology")
    assert parsed1["subreddit"] == "technology"
    parsed2 = parse_reddit_target("https://www.reddit.com/r/science/comments/abc123/example_title/")
    assert parsed2["post_id"] == "abc123"
    
    source = RedditSource(target="technology")
    meta = source.fetch_meta()
    assert meta["source"] == "reddit"
    print(f"Reddit meta fetched: {meta['title']}")
    
    comments = source.fetch_comments(limit=3)
    print(f"Fetched {len(comments)} Reddit comments (first: {comments[0]['author'] if comments else 'None'})")
    assert len(comments) > 0, "No Reddit comments returned"
    print("Reddit Source: PASSED [OK]")

def test_flask_endpoints():
    print("\n--- 4. Testing Flask API Routes ---")
    from web.app import app
    client = app.test_client()
    
    # Test index
    r_index = client.get("/")
    assert r_index.status_code == 200
    
    # Test model_info
    r_info = client.get("/api/model_info")
    assert r_info.status_code == 200
    assert r_info.json["supports_lime"] == True
    
    # Test bias_metrics
    r_bias = client.get("/api/bias_metrics")
    assert r_bias.status_code == 200
    assert len(r_bias.json["identities"]) == 9
    assert "overall" in r_bias.json
    print(f"Bias Metrics: 9 identity groups verified, mean BPSN gain: {r_bias.json['overall']['bpsn_gain_pct']}%")
    
    # Test explain API
    r_explain = client.post("/api/explain", json={"text": "I love this community", "model": "transformer"})
    assert r_explain.status_code == 200
    assert "features" in r_explain.json
    
    print("Flask Endpoints: PASSED [OK]")

if __name__ == "__main__":
    test_transformer_scorer()
    test_lime_explainability()
    test_reddit_source()
    test_flask_endpoints()
    print("\n==========================================")
    print("ALL END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
