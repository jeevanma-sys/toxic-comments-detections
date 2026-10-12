"""
train_model.py

Offline training pipeline:
  1. Load Jigsaw dataset
  2. Clean + vectorize text (TF-IDF)
  3. Train a baseline classifier (Logistic Regression)
  4. Evaluate accuracy AND fairness (subgroup/BPSN/BNSP AUC, FPR/FNR gaps)
  5. Save model + vectorizer to /models

Run:
  python src/train_model.py --data data/train.csv --sample_size 50000
"""

import argparse
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

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


def main(args):
    print(f"Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data)

    available_identity_cols = [c for c in IDENTITY_COLS if c in df.columns]
    df = df.dropna(subset=["comment_text", "target"]).reset_index(drop=True)

    if args.sample_size:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42).reset_index(drop=True)

    print(f"Rows used: {len(df)}")
    print(f"Identity columns found: {available_identity_cols}")

    df["target_bin"] = (df["target"] >= 0.5).astype(int)
    for col in available_identity_cols:
        df[col] = df[col].fillna(0)

    print("Cleaning text ...")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df["clean_text"], df["target_bin"], df.index,
        test_size=0.2, random_state=42, stratify=df["target_bin"]
    )

    print("Vectorizing (TF-IDF) ...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression ...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train_vec, y_train)

    print("\n=== Overall performance ===")
    y_pred_prob = clf.predict_proba(X_test_vec)[:, 1]
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)
    overall_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Overall AUC: {overall_auc:.4f}")
    print(classification_report(y_test, y_pred_bin))

    print("\n=== Fairness evaluation ===")
    test_df = df.loc[idx_test].copy()
    test_df["pred_prob"] = y_pred_prob
    test_df["pred_bin"] = y_pred_bin

    if available_identity_cols:
        bias_df = compute_subgroup_aucs(test_df, available_identity_cols)
        print("\nPer-identity bias AUCs (lower = more biased):")
        print(bias_df.to_string(index=False))

        summary = final_combined_score(bias_df, overall_auc)
        print("\nCombined fairness score:")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")

        fpr_fnr_df = group_fpr_fnr(test_df, available_identity_cols)
        print("\nFalse Positive / False Negative rate per identity:")
        print(fpr_fnr_df.to_string(index=False))

        bias_df.to_csv(args.out_dir + "/bias_report.csv", index=False)
        fpr_fnr_df.to_csv(args.out_dir + "/fpr_fnr_report.csv", index=False)
    else:
        print("No identity columns found in dataset - skipping bias breakdown.")

    joblib.dump(clf, f"{args.out_dir}/toxicity_model.pkl")
    joblib.dump(vectorizer, f"{args.out_dir}/tfidf_vectorizer.pkl")
    print(f"\nSaved model + vectorizer to {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.csv")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--sample_size", type=int, default=None)
    args = parser.parse_args()
    main(args)