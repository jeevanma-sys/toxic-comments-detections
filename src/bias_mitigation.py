"""
bias_mitigation.py

Two mitigation techniques, compared against the baseline:

  1. REWEIGHING (pre-processing):
     Upweight non-toxic comments within identity subgroups so the
     model is penalized more for wrongly flagging them as toxic.

  2. THRESHOLD ADJUSTMENT (post-processing, manual):
     Instead of one global 0.5 cutoff, find per-group thresholds that
     bring false positive rates closer together across identity groups.

Run:
  python src/bias_mitigation.py --data data/train.csv --sample_size 50000
"""

import argparse
import re
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import sys, os
sys.path.append(os.path.dirname(__file__))
from fairness_utils import (
    compute_subgroup_aucs,
    final_combined_score,
    group_fpr_fnr,
)

IDENTITY_COLS = [
    "male", "female", "homosexual_gay_or_lesbian", "christian", "jewish",
    "muslim", "black", "white", "psychiatric_or_mental_illness",
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sample_weights(df, identity_cols, y_col="target_bin", identity_threshold=0.5):
    weights = np.ones(len(df))
    overall_pos_rate = df[y_col].mean()

    for identity in identity_cols:
        mask = df[identity] >= identity_threshold
        if mask.sum() < 10:
            continue
        subgroup_pos_rate = df.loc[mask, y_col].mean()
        if subgroup_pos_rate in (0, 1) or np.isnan(subgroup_pos_rate):
            continue

        nontoxic_mask = mask & (df[y_col] == 0)
        expected_nontoxic_share = 1 - overall_pos_rate
        actual_nontoxic_share = 1 - subgroup_pos_rate
        if actual_nontoxic_share > 0:
            boost = expected_nontoxic_share / actual_nontoxic_share
            weights[nontoxic_mask.values] *= max(1.0, boost)

    return weights


def evaluate(df, y_true_col, prob_col, bin_col, identity_cols, label):
    print(f"\n=== {label} ===")
    overall_auc = roc_auc_score(df[y_true_col], df[prob_col])
    print(f"Overall AUC: {overall_auc:.4f}")

    bias_df = compute_subgroup_aucs(df, identity_cols, y_true_col=y_true_col, y_pred_col=prob_col)
    print(bias_df.to_string(index=False))

    summary = final_combined_score(bias_df, overall_auc)
    print("Combined fairness score:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    fpr_df = group_fpr_fnr(df, identity_cols, y_true_col=y_true_col, y_pred_bin_col=bin_col)
    print("FPR / FNR per identity:")
    print(fpr_df.to_string(index=False))

    return bias_df, fpr_df, summary


def main(args):
    print(f"Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data)
    available_identity_cols = [c for c in IDENTITY_COLS if c in df.columns]
    df = df.dropna(subset=["comment_text", "target"]).reset_index(drop=True)

    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42).reset_index(drop=True)

    df["target_bin"] = (df["target"] >= 0.5).astype(int)
    for col in available_identity_cols:
        df[col] = df[col].fillna(0)
    df["clean_text"] = df["comment_text"].apply(clean_text)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df["clean_text"], df["target_bin"], df.index,
        test_size=0.2, random_state=42, stratify=df["target_bin"]
    )
    train_df = df.loc[idx_train].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    X_train_vec = vectorizer.fit_transform(train_df["clean_text"])
    X_test_vec = vectorizer.transform(test_df["clean_text"])

    # ---------- BASELINE (no mitigation) ----------
    print("\nTraining BASELINE model (no mitigation) ...")
    baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    baseline_clf.fit(X_train_vec, train_df["target_bin"])

    test_df["baseline_prob"] = baseline_clf.predict_proba(X_test_vec)[:, 1]
    test_df["baseline_bin"] = (test_df["baseline_prob"] >= 0.5).astype(int)

    baseline_bias_df, baseline_fpr_df, baseline_summary = evaluate(
        test_df, "target_bin", "baseline_prob", "baseline_bin",
        available_identity_cols, "BASELINE (before mitigation)"
    )

    # ---------- MITIGATED: reweighing ----------
    print("\nTraining MITIGATED model (reweighing) ...")
    sample_weights = build_sample_weights(train_df, available_identity_cols, y_col="target_bin")
    mitigated_clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    mitigated_clf.fit(X_train_vec, train_df["target_bin"], sample_weight=sample_weights)

    test_df["mitigated_prob"] = mitigated_clf.predict_proba(X_test_vec)[:, 1]
    test_df["mitigated_bin"] = (test_df["mitigated_prob"] >= 0.5).astype(int)

    mitigated_bias_df, mitigated_fpr_df, mitigated_summary = evaluate(
        test_df, "target_bin", "mitigated_prob", "mitigated_bin",
        available_identity_cols, "MITIGATED (after reweighing)"
    )

    # ---------- POST-PROCESSING: manual per-group threshold adjustment ----------
    worst_identity = baseline_bias_df.sort_values("bpsn_auc").iloc[0]["identity"]
    print(f"\nApplying per-group threshold adjustment using sensitive feature: {worst_identity}")

    target_fpr = baseline_fpr_df["fpr"].median()  # aim every group toward the median FPR
    print(f"Target FPR (median across groups): {target_fpr:.4f}")

    sensitive_test = (test_df[worst_identity] >= 0.5).astype(int).values
    postproc_bin = np.zeros(len(test_df), dtype=int)

    for group_value in [0, 1]:
        mask = sensitive_test == group_value
        if mask.sum() < 10:
            postproc_bin[mask] = test_df.loc[mask, "mitigated_bin"]
            continue
        y_true_group = test_df.loc[mask, "target_bin"].values
        y_prob_group = test_df.loc[mask, "mitigated_prob"].values

        fpr, tpr, thresholds = roc_curve(y_true_group, y_prob_group)
        idx = np.argmin(np.abs(fpr - target_fpr))
        chosen_threshold = thresholds[idx]

        postproc_bin[mask] = (y_prob_group >= chosen_threshold).astype(int)
        print(f"  Group={worst_identity}={group_value}: chosen threshold={chosen_threshold:.3f}")

    test_df["postproc_bin"] = postproc_bin

    postproc_fpr_df = group_fpr_fnr(
        test_df, available_identity_cols, y_true_col="target_bin", y_pred_bin_col="postproc_bin"
    )
    print("\nFPR/FNR AFTER threshold adjustment (post-processing):")
    print(postproc_fpr_df.to_string(index=False))

    # ---------- Save comparison tables ----------
    comparison = baseline_bias_df.merge(
        mitigated_bias_df, on="identity", suffixes=("_baseline", "_mitigated")
    )
    comparison.to_csv(f"{args.out_dir}/before_after_comparison.csv", index=False)

    fpr_comparison = baseline_fpr_df.merge(
        mitigated_fpr_df, on="identity", suffixes=("_baseline", "_mitigated")
    ).merge(
        postproc_fpr_df[["identity", "fpr", "fnr"]].rename(
            columns={"fpr": "fpr_postproc", "fnr": "fnr_postproc"}), on="identity"
    )
    fpr_comparison.to_csv(f"{args.out_dir}/fpr_before_after_comparison.csv", index=False)

    print("\nSaved comparison tables to models/before_after_comparison.csv "
          "and models/fpr_before_after_comparison.csv")
    print("\nSummary (overall AUC, subgroup GM AUC):")
    print(f"  Baseline : overall={baseline_summary['overall_auc']:.4f}, "
          f"subgroup_gm={baseline_summary['subgroup_gm_auc']:.4f}")
    print(f"  Mitigated: overall={mitigated_summary['overall_auc']:.4f}, "
          f"subgroup_gm={mitigated_summary['subgroup_gm_auc']:.4f}")

    joblib.dump(mitigated_clf, f"{args.out_dir}/toxicity_model_mitigated.pkl")
    joblib.dump(vectorizer, f"{args.out_dir}/tfidf_vectorizer_mitigated.pkl")
    print(f"Saved mitigated model to {args.out_dir}/toxicity_model_mitigated.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.csv")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    main(args)