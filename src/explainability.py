"""
src/explainability.py

LIME (Local Interpretable Model-agnostic Explanations) Module:
  - Generates token-level attribution weights explaining WHY a model
    predicted a comment as toxic or non-toxic.
  - Supports both Transformer Context-Aware Scorer and TF-IDF models.
  - Returns structured feature weights, token-level visual heatmaps,
    and HTML representations for frontends.

Usage:
  from explainability import get_explainer
  explainer = get_explainer()
  explanation = explainer.explain_comment("You are an idiot", model_type="transformer")
"""

import os
import re
import html
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformer_scorer import (
    get_transformer_scorer,
    clean_text_bow,
    clean_text_simple,
    MITIGATED_MODEL_PATH,
    MITIGATED_VEC_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_VEC_PATH,
)


class ToxicityExplainer:
    """
    Manages LIME explanations for toxicity classification across
    different model architectures (Transformer vs TF-IDF).
    """

    def __init__(self):
        self.explainer = LimeTextExplainer(
            class_names=["non-toxic", "toxic"],
            split_expression=r"\W+",
            bow=False,  # preserves order in perturbed instances
        )
        self.transformer_scorer = None
        self.mitigated_model = None
        self.mitigated_vec = None
        self.baseline_model = None
        self.baseline_vec = None

    def _get_transformer_scorer(self):
        if self.transformer_scorer is None:
            self.transformer_scorer = get_transformer_scorer()
        return self.transformer_scorer

    def _get_mitigated_model(self):
        if self.mitigated_model is None and os.path.exists(MITIGATED_MODEL_PATH):
            self.mitigated_model = joblib.load(MITIGATED_MODEL_PATH)
            self.mitigated_vec = joblib.load(MITIGATED_VEC_PATH)
        return self.mitigated_model, self.mitigated_vec

    def _get_baseline_model(self):
        if self.baseline_model is None and os.path.exists(BASELINE_MODEL_PATH):
            self.baseline_model = joblib.load(BASELINE_MODEL_PATH)
            self.baseline_vec = joblib.load(BASELINE_VEC_PATH)
        return self.baseline_model, self.baseline_vec

    def _get_predict_fn(self, model_type: str):
        """Returns the predict_proba callable corresponding to model_type."""
        model_type = model_type.lower()

        if model_type in ["transformer", "bert", "deep_learning"]:
            scorer = self._get_transformer_scorer()
            return scorer.predict_proba

        elif model_type in ["mitigated", "tfidf_mitigated"]:
            clf, vec = self._get_mitigated_model()
            if clf is not None and vec is not None:
                def predict_mitigated(texts):
                    bow_texts = [clean_text_bow(t) for t in texts]
                    v = vec.transform(bow_texts)
                    return clf.predict_proba(v)
                return predict_mitigated

        elif model_type in ["baseline", "tfidf_baseline"]:
            clf, vec = self._get_baseline_model()
            if clf is not None and vec is not None:
                def predict_baseline(texts):
                    bow_texts = [clean_text_bow(t) for t in texts]
                    v = vec.transform(bow_texts)
                    return clf.predict_proba(v)
                return predict_baseline

        # Default fallback to transformer
        scorer = self._get_transformer_scorer()
        return scorer.predict_proba

    def explain_comment(
        self,
        text: str,
        model_type: str = "transformer",
        num_features: int = 8,
        num_samples: int = 150,
    ) -> dict:
        """
        Explain a single comment using LIME.
        Returns:
          - prediction probabilities
          - word feature attribution list with direction and weights
          - inline highlighted HTML visualization
        """
        if not text or not text.strip():
            return {
                "text": text,
                "error": "Empty text provided",
                "features": [],
                "prediction": {"toxicity_prob": 0.0, "is_toxic": False},
            }

        predict_fn = self._get_predict_fn(model_type)

        # Baseline prediction on original text
        raw_prob = predict_fn([text])[0]
        p_nontoxic = float(raw_prob[0])
        p_toxic = float(raw_prob[1])
        is_toxic = p_toxic >= 0.5

        # Run LIME perturbation and explanation for the 'toxic' class (index 1)
        exp = self.explainer.explain_instance(
            text,
            predict_fn,
            labels=(1,),
            num_features=num_features,
            num_samples=num_samples,
        )

        raw_weights = exp.as_list(label=1)
        # raw_weights is a list of tuples: [('idiot', 0.52), ('awesome', -0.31), ...]

        features = []
        max_abs = max([abs(w) for _, w in raw_weights], default=1.0)
        if max_abs == 0:
            max_abs = 1.0

        for word, weight in raw_weights:
            direction = "toxic" if weight > 0 else "non-toxic"
            normalized_impact = abs(weight) / max_abs
            # Red color for toxic-increasing words, green color for non-toxic / calming words
            if direction == "toxic":
                color = f"rgba(225, 87, 89, {0.25 + 0.75 * normalized_impact:.2f})"
            else:
                color = f"rgba(89, 161, 79, {0.25 + 0.75 * normalized_impact:.2f})"

            features.append({
                "word": word,
                "weight": round(weight, 4),
                "direction": direction,
                "abs_weight": round(abs(weight), 4),
                "color": color,
            })

        # Generate HTML snippet with inline highlights
        highlighted_html = self._generate_highlighted_html(text, features)

        return {
            "text": text,
            "model_type": model_type,
            "prediction": {
                "toxicity_prob": round(p_toxic, 4),
                "non_toxicity_prob": round(p_nontoxic, 4),
                "is_toxic": is_toxic,
                "label": "toxic" if is_toxic else "non-toxic",
            },
            "features": features,
            "highlighted_html": highlighted_html,
        }

    def _generate_highlighted_html(self, text: str, features: list[dict]) -> str:
        """Create rich HTML with spans highlighting impactful words according to LIME weights."""
        feature_map = {f["word"].lower(): f for f in features}

        # Tokenize by keeping word boundaries and punctuation
        tokens = re.split(r"(\W+)", text)
        highlighted_tokens = []

        for token in tokens:
            lower = token.lower()
            if lower in feature_map:
                f = feature_map[lower]
                direction_label = "Increases toxicity" if f["direction"] == "toxic" else "Reduces toxicity"
                badge = (
                    f'<span class="lime-token" '
                    f'style="background-color:{f["color"]}; padding:2px 5px; margin:0 1px; border-radius:4px; font-weight:600; cursor:pointer;" '
                    f'title="{direction_label} (weight: {f["weight"]:+.3f})">'
                    f'{html.escape(token)}'
                    f'</span>'
                )
                highlighted_tokens.append(badge)
            else:
                highlighted_tokens.append(html.escape(token))

        return "".join(highlighted_tokens)

    def compare_explanations(self, text: str, num_features: int = 6) -> dict:
        """Run LIME on both Transformer and TF-IDF Mitigated models to contrast word attributions."""
        trans_exp = self.explain_comment(text, model_type="transformer", num_features=num_features)
        tfidf_exp = self.explain_comment(text, model_type="mitigated", num_features=num_features)

        return {
            "text": text,
            "transformer": trans_exp,
            "tfidf_mitigated": tfidf_exp,
        }


# Singleton instance
_global_explainer = None


def get_explainer() -> ToxicityExplainer:
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = ToxicityExplainer()
    return _global_explainer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="You are an absolute idiot and nobody likes you")
    parser.add_argument("--model", type=str, default="transformer")
    args = parser.parse_args()

    explainer = get_explainer()
    print(f"\n--- Running LIME Explanation on model={args.model} ---")
    print(f"Target: '{args.text}'")
    result = explainer.explain_comment(args.text, model_type=args.model, num_features=6, num_samples=100)

    pred = result["prediction"]
    print(f"\nPrediction: {pred['label'].upper()} (Prob: {pred['toxicity_prob']:.4f})")
    print("\nTop Contributing Word Features:")
    for f in result["features"]:
        bar = "=" * max(1, int(f["abs_weight"] * 20))
        direction_sym = "[+TOXIC]" if f["direction"] == "toxic" else "[-CALM]"
        print(f"  {f['word']:<15} {direction_sym:<10} {f['weight']:>+7.4f}  {bar}")
