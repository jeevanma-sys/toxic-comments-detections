"""
src/transformer_scorer.py

Transformer-Based Context-Aware Toxicity Scorer:
  - Leverages pre-trained contextual deep transformer architecture (Unitary Toxic-BERT).
  - Understands nuances that bag-of-words (TF-IDF) models fail on:
      1. Negation handling (e.g. "not bad at all", "you are not an idiot")
      2. Benign identity mentions / reclamation (e.g. "proud gay woman", "as a Muslim, I love peace")
      3. Benign word context (e.g. "the kill streak in this game is epic")
  - Evaluates multi-dimensional toxicity:
      - toxic, severe_toxic, obscene, threat, insult, identity_hate
  - Provides scikit-learn / LIME compatible `predict_proba(texts)` returning [P(non-toxic), P(toxic)].
  - Provides graceful fallback to TF-IDF mitigated model if transformer is unavailable.

Usage:
  from transformer_scorer import TransformerToxicityScorer
  scorer = TransformerToxicityScorer()
  result = scorer.score_text("I love this community!")
  # result -> {'toxicity_prob': 0.002, 'is_toxic': False, 'label': 'non-toxic', ...}
"""

import os
import re
import joblib
import numpy as np

# Gold-standard benchmark model from Unitary AI trained specifically on Jigsaw Toxicity data
TRANSFORMER_MODEL_NAME = "unitary/toxic-bert"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MITIGATED_MODEL_PATH = os.path.join(BASE_DIR, "models", "toxicity_model_mitigated.pkl")
MITIGATED_VEC_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer_mitigated.pkl")
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "toxicity_model.pkl")
BASELINE_VEC_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

IDENTITY_KEYWORDS = {
    "female": ["woman", "women", "girl", "she", "her", "female"],
    "male": ["man", "men", "boy", "he", "him", "male"],
    "muslim": ["muslim", "islam", "islamic"],
    "christian": ["christian", "christianity", "catholic"],
    "jewish": ["jewish", "jew", "judaism"],
    "homosexual_gay_or_lesbian": ["gay", "lesbian", "homosexual", "lgbt", "queer"],
    "black": ["black person", "black people", "african american"],
    "white": ["white person", "white people"],
    "psychiatric_or_mental_illness": ["mental illness", "depression", "schizophrenia", "psychiatric"],
}


def clean_text_simple(text: str) -> str:
    """Minimal cleaning that preserves sentence structure and negation words for Transformers."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_bow(text: str) -> str:
    """Aggressive lowercase cleaning used by traditional bag-of-words / TF-IDF models."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_identities(text: str) -> list[str]:
    """Detect sensitive identity mentions for fairness tracking."""
    text_lower = text.lower()
    flags = []
    for identity, keywords in IDENTITY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(k)}\b", text_lower) for k in keywords):
            flags.append(identity)
    return flags


class TransformerToxicityScorer:
    """
    Context-Aware Toxicity Scorer backed by a Hugging Face Transformer pipeline.
    Falls back gracefully to TF-IDF mitigated model if Transformer loading fails.
    """

    def __init__(self, model_name: str = TRANSFORMER_MODEL_NAME, threshold: float = 0.5):
        self.model_name = model_name
        self.threshold = threshold
        self.pipeline = None
        self.fallback_model = None
        self.fallback_vectorizer = None
        self.backend = "transformer"
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            print(f"[TransformerScorer] Initializing Transformer pipeline ({self.model_name}) ...")
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self.backend = "transformer"
            print(f"[TransformerScorer] Transformer pipeline loaded successfully on CPU.")
        except Exception as e:
            print(f"[TransformerScorer] Warning: Could not load transformer ({e}). Using TF-IDF mitigated fallback.")
            self._load_fallback()

    def _load_fallback(self):
        try:
            if os.path.exists(MITIGATED_MODEL_PATH) and os.path.exists(MITIGATED_VEC_PATH):
                self.fallback_model = joblib.load(MITIGATED_MODEL_PATH)
                self.fallback_vectorizer = joblib.load(MITIGATED_VEC_PATH)
                self.backend = "tfidf_mitigated"
                print(f"[TransformerScorer] Loaded TF-IDF mitigated fallback model.")
            elif os.path.exists(BASELINE_MODEL_PATH) and os.path.exists(BASELINE_VEC_PATH):
                self.fallback_model = joblib.load(BASELINE_MODEL_PATH)
                self.fallback_vectorizer = joblib.load(BASELINE_VEC_PATH)
                self.backend = "tfidf_baseline"
                print(f"[TransformerScorer] Loaded TF-IDF baseline fallback model.")
            else:
                self.backend = "heuristic"
                print("[TransformerScorer] Warning: No offline model found. Using heuristic fallback.")
        except Exception as e:
            print(f"[TransformerScorer] Error loading fallback: {e}")
            self.backend = "heuristic"

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Scikit-learn / LIME compatible probability prediction:
        Returns numpy array of shape (N, 2):
          column 0: P(non-toxic)
          column 1: P(toxic)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 2))

        cleaned_texts = [clean_text_simple(t) or "empty" for t in texts]

        if self.pipeline is not None and self.backend == "transformer":
            try:
                results = self.pipeline(cleaned_texts)
                probs = np.zeros((len(texts), 2))
                for i, res in enumerate(results):
                    # res is a list of dicts: [{'label': 'toxic', 'score': ...}, {'label': 'insult', ...}]
                    score_map = {item["label"].lower(): float(item["score"]) for item in res}
                    p_toxic = score_map.get("toxic", 0.0)
                    # If high insult or severe_toxic is detected, adjust toxic prob if necessary
                    max_sub_toxic = max(
                        score_map.get("insult", 0.0),
                        score_map.get("severe_toxic", 0.0),
                        score_map.get("obscene", 0.0),
                        score_map.get("threat", 0.0),
                        score_map.get("identity_hate", 0.0),
                    )
                    effective_p_toxic = max(p_toxic, max_sub_toxic)
                    p_nontoxic = max(0.0, 1.0 - effective_p_toxic)

                    # Normalize sum to 1.0
                    total = effective_p_toxic + p_nontoxic
                    if total > 0:
                        probs[i, 0] = p_nontoxic / total
                        probs[i, 1] = effective_p_toxic / total
                    else:
                        probs[i, 0] = 1.0
                        probs[i, 1] = 0.0
                return probs
            except Exception as e:
                print(f"[TransformerScorer] Transformer inference error: {e}. Falling back to TF-IDF.")

        # Fallback to TF-IDF model
        if self.fallback_model is not None and self.fallback_vectorizer is not None:
            bow_texts = [clean_text_bow(t) for t in texts]
            vec = self.fallback_vectorizer.transform(bow_texts)
            return self.fallback_model.predict_proba(vec)

        # Simple heuristic fallback
        probs = np.zeros((len(texts), 2))
        toxic_triggers = {"idiot", "stupid", "moron", "hate", "kill", "dumb", "ugly"}
        for i, t in enumerate(texts):
            words = set(re.findall(r"\w+", t.lower()))
            is_bad = bool(words & toxic_triggers)
            p_toxic = 0.85 if is_bad else 0.05
            probs[i, 0] = 1.0 - p_toxic
            probs[i, 1] = p_toxic
        return probs

    def score_text(self, text: str) -> dict:
        """Score a single comment with rich context metadata and subcategories."""
        probs = self.predict_proba([text])[0]
        p_nontoxic = float(probs[0])
        p_toxic = float(probs[1])
        is_toxic = p_toxic >= self.threshold
        identities = detect_identities(text)

        subcategories = {}
        if self.pipeline is not None and self.backend == "transformer":
            try:
                res = self.pipeline(clean_text_simple(text) or "empty")[0]
                for item in res:
                    subcategories[item["label"].lower()] = round(float(item["score"]), 4)
            except Exception:
                pass

        # Detect notable context cues (e.g. negation or benign identity mention)
        context_notes = []
        text_lower = text.lower()
        if any(neg in text_lower for neg in ["not ", "n't ", "never ", "no one ", "neither "]):
            context_notes.append("Contains negation syntax")
        if identities:
            if not is_toxic:
                context_notes.append(f"Benign identity assertion ({', '.join(identities)})")
            else:
                context_notes.append(f"Identity-targeted risk ({', '.join(identities)})")

        return {
            "text": text,
            "toxicity_prob": round(p_toxic, 4),
            "non_toxicity_prob": round(p_nontoxic, 4),
            "is_toxic": is_toxic,
            "label": "toxic" if is_toxic else "non-toxic",
            "confidence": round(max(p_toxic, p_nontoxic), 4),
            "backend": self.backend,
            "model_name": self.model_name if self.backend == "transformer" else "TF-IDF Mitigated",
            "identity_flags": identities,
            "context_notes": context_notes,
            "subcategories": subcategories,
        }

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Score a list of comments efficiently."""
        if not texts:
            return []
        probs = self.predict_proba(texts)
        results = []
        for i, text in enumerate(texts):
            p_toxic = float(probs[i, 1])
            p_nontoxic = float(probs[i, 0])
            is_toxic = p_toxic >= self.threshold
            identities = detect_identities(text)
            results.append({
                "text": text,
                "toxicity_prob": round(p_toxic, 4),
                "non_toxicity_prob": round(p_nontoxic, 4),
                "is_toxic": is_toxic,
                "label": "toxic" if is_toxic else "non-toxic",
                "backend": self.backend,
                "identity_flags": identities,
            })
        return results

    def compare_with_bow(self, text: str) -> dict:
        """
        Compare Transformer Context-Aware prediction directly against
        the TF-IDF Mitigated and Baseline Bag-of-Words models.
        """
        trans_res = self.score_text(text)
        comparison = {
            "text": text,
            "transformer": trans_res,
            "bow_mitigated": None,
            "bow_baseline": None,
        }

        # Mitigated TF-IDF
        if os.path.exists(MITIGATED_MODEL_PATH) and os.path.exists(MITIGATED_VEC_PATH):
            try:
                mit_model = joblib.load(MITIGATED_MODEL_PATH)
                mit_vec = joblib.load(MITIGATED_VEC_PATH)
                bow_cleaned = clean_text_bow(text)
                v = mit_vec.transform([bow_cleaned])
                p_mit = float(mit_model.predict_proba(v)[0, 1])
                comparison["bow_mitigated"] = {
                    "toxicity_prob": round(p_mit, 4),
                    "is_toxic": p_mit >= self.threshold,
                }
            except Exception as e:
                comparison["bow_mitigated"] = {"error": str(e)}

        # Baseline TF-IDF
        if os.path.exists(BASELINE_MODEL_PATH) and os.path.exists(BASELINE_VEC_PATH):
            try:
                base_model = joblib.load(BASELINE_MODEL_PATH)
                base_vec = joblib.load(BASELINE_VEC_PATH)
                bow_cleaned = clean_text_bow(text)
                v = base_vec.transform([bow_cleaned])
                p_base = float(base_model.predict_proba(v)[0, 1])
                comparison["bow_baseline"] = {
                    "toxicity_prob": round(p_base, 4),
                    "is_toxic": p_base >= self.threshold,
                }
            except Exception as e:
                comparison["bow_baseline"] = {"error": str(e)}

        return comparison


# Global singleton instance for fast reuse across threads
_global_scorer = None


def get_transformer_scorer() -> TransformerToxicityScorer:
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = TransformerToxicityScorer()
    return _global_scorer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run context-aware comparison tests")
    parser.add_argument("--text", type=str, default=None, help="Score a single custom text")
    args = parser.parse_args()

    scorer = get_transformer_scorer()

    if args.text:
        res = scorer.compare_with_bow(args.text)
        print("\n=== Single Text Evaluation ===")
        print(f"Text: '{res['text']}'")
        print(f"Transformer Context-Aware: prob={res['transformer']['toxicity_prob']:.4f} toxic={res['transformer']['is_toxic']}")
        if res["bow_mitigated"]:
            print(f"TF-IDF Mitigated BoW   : prob={res['bow_mitigated']['toxicity_prob']:.4f} toxic={res['bow_mitigated']['is_toxic']}")
        if res["bow_baseline"]:
            print(f"TF-IDF Baseline BoW    : prob={res['bow_baseline']['toxicity_prob']:.4f} toxic={res['bow_baseline']['is_toxic']}")
    else:
        print("\n=== Context-Awareness Benchmark Test Cases ===")
        test_cases = [
            ("I am a proud gay woman and I love my community", "Benign identity assertion"),
            ("You are not an idiot, you did an amazing job!", "Negation of insult"),
            ("The kill streak in this video game was incredible", "Benign gaming vernacular"),
            ("You are a stupid idiot and nobody likes you", "Explicit toxic insult"),
            ("As a Muslim, peace and tolerance are core to my faith", "Benign religious mention"),
            ("I strongly disagree with your political opinion, but I respect you", "Civil debate"),
        ]
        print(f"{'Comment':<55} | {'Case Type':<25} | {'Transformer':<12} | {'TF-IDF Mit':<12}")
        print("-" * 110)
        for t, case_type in test_cases:
            cmp = scorer.compare_with_bow(t)
            t_prob = cmp["transformer"]["toxicity_prob"]
            b_prob = cmp["bow_mitigated"]["toxicity_prob"] if cmp["bow_mitigated"] else 0.0
            print(f"{t[:53]:<55} | {case_type:<25} | {t_prob:<12.4f} | {b_prob:<12.4f}")
