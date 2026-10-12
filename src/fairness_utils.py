"""
fairness_utils.py

Implements the bias/fairness evaluation metrics used in the
Jigsaw "Unintended Bias in Toxicity Classification" benchmark,
plus standard group-fairness metrics.

Core idea:
  For each identity subgroup (e.g. "muslim", "female", "black"),
  we split test comments into 4 buckets:

    Subgroup Positive (SP): mentions identity, truly toxic
    Subgroup Negative (SN): mentions identity, truly non-toxic
    Background Positive (BP): does NOT mention identity, truly toxic
    Background Negative (BN): does NOT mention identity, truly non-toxic

  From these we compute 3 AUC scores per identity:

    Subgroup AUC : AUC computed ONLY on comments mentioning the identity
                   (SP vs SN). Low score => model can't tell toxic vs
                   non-toxic for this identity specifically.

    BPSN AUC     : AUC on (BP vs SN) i.e. background-toxic vs
                   subgroup-non-toxic. Low score => model confuses
                   non-toxic identity mentions with toxic content
                   ("false positive bias" - flags identity mentions
                   as toxic even when they're not).

    BNSP AUC     : AUC on (BN vs SP) i.e. background-non-toxic vs
                   subgroup-toxic. Low score => model fails to catch
                   real toxicity when it targets/mentions this identity
                   ("false negative bias" - under-protects the group).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix


def compute_subgroup_aucs(df, identity_cols, y_true_col="target_bin",
                           y_pred_col="pred_prob", identity_threshold=0.5,
                           toxic_threshold=0.5):
    """
    df must contain: y_true_col (0/1), y_pred_col (predicted probability),
    and one column per identity in identity_cols with values in [0,1]
    (fraction of raters who said the comment mentions that identity).

    Returns a DataFrame: one row per identity, columns =
    [identity, subgroup_size, subgroup_auc, bpsn_auc, bnsp_auc]
    """
    results = []

    for identity in identity_cols:
        subgroup_mask = df[identity] >= identity_threshold
        background_mask = ~subgroup_mask

        subgroup = df[subgroup_mask]
        background = df[background_mask]

        if subgroup[y_true_col].nunique() < 2 or len(subgroup) < 10:
            results.append({
                "identity": identity,
                "subgroup_size": len(subgroup),
                "subgroup_auc": np.nan,
                "bpsn_auc": np.nan,
                "bnsp_auc": np.nan,
            })
            continue

        # 1. Subgroup AUC
        subgroup_auc = roc_auc_score(subgroup[y_true_col], subgroup[y_pred_col])

        # 2. BPSN AUC: background-positive vs subgroup-negative
        bp = background[background[y_true_col] == 1]
        sn = subgroup[subgroup[y_true_col] == 0]
        bpsn_set = pd.concat([bp, sn])
        bpsn_auc = (roc_auc_score(bpsn_set[y_true_col], bpsn_set[y_pred_col])
                    if bpsn_set[y_true_col].nunique() == 2 else np.nan)

        # 3. BNSP AUC: background-negative vs subgroup-positive
        bn = background[background[y_true_col] == 0]
        sp = subgroup[subgroup[y_true_col] == 1]
        bnsp_set = pd.concat([bn, sp])
        bnsp_auc = (roc_auc_score(bnsp_set[y_true_col], bnsp_set[y_pred_col])
                    if bnsp_set[y_true_col].nunique() == 2 else np.nan)

        results.append({
            "identity": identity,
            "subgroup_size": len(subgroup),
            "subgroup_auc": subgroup_auc,
            "bpsn_auc": bpsn_auc,
            "bnsp_auc": bnsp_auc,
        })

    return pd.DataFrame(results).sort_values("subgroup_auc")


def generalized_mean_bias_auc(bias_df, col, power=-5):
    """
    Combines per-identity AUCs into a single 'worst-case weighted' score.
    Negative power heavily penalizes identities with the LOWEST scores
    (i.e. the identities the model is most biased against).
    """
    scores = bias_df[col].dropna().values
    scores = np.clip(scores, 1e-6, 1.0)
    return np.power(np.mean(np.power(scores, power)), 1.0 / power)


def final_combined_score(bias_df, overall_auc, weight_overall=0.25):
    """
    Jigsaw-style final metric: weighted average of overall AUC and the
    3 generalized-mean bias AUCs.
    """
    sub_gm = generalized_mean_bias_auc(bias_df, "subgroup_auc")
    bpsn_gm = generalized_mean_bias_auc(bias_df, "bpsn_auc")
    bnsp_gm = generalized_mean_bias_auc(bias_df, "bnsp_auc")

    bias_component = np.mean([sub_gm, bpsn_gm, bnsp_gm])
    score = weight_overall * overall_auc + (1 - weight_overall) * bias_component
    return {
        "overall_auc": overall_auc,
        "subgroup_gm_auc": sub_gm,
        "bpsn_gm_auc": bpsn_gm,
        "bnsp_gm_auc": bnsp_gm,
        "final_combined_score": score,
    }


def group_fpr_fnr(df, identity_cols, y_true_col="target_bin",
                   y_pred_bin_col="pred_bin", identity_threshold=0.5):
    """
    False Positive Rate and False Negative Rate per identity subgroup.
    This is usually the clearest chart for your report/presentation:
    "comments mentioning X are flagged toxic Y% more often."
    """
    rows = []
    for identity in identity_cols:
        mask = df[identity] >= identity_threshold
        sub = df[mask]
        if len(sub) < 10:
            continue
        tn, fp, fn, tp = confusion_matrix(
            sub[y_true_col], sub[y_pred_bin_col], labels=[0, 1]
        ).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        rows.append({"identity": identity, "n": len(sub), "fpr": fpr, "fnr": fnr})
    return pd.DataFrame(rows).sort_values("fpr", ascending=False)