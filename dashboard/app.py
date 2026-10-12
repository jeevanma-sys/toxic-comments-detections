"""
dashboard/app.py

Professional Toxic Comment & Fairness Monitor:
  - Tab 1: 🔴 Live Stream Monitor (YouTube & Reddit live feeds, filters, fairness KPIs)
  - Tab 2: 🔬 LIME Explainability Panel (word impact weights, token heatmaps, model comparison)
  - Tab 3: ⚖️ Bias & Fairness Dashboard (BPSN/BNSP AUC, FPR parity before vs after mitigation)

Run:
  streamlit run dashboard/app.py
"""

import os
import sys
import html
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add project src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from transformer_scorer import get_transformer_scorer, detect_identities
from explainability import get_explainer
from reddit_source import RedditSource

CSV_PATH = os.path.join(BASE_DIR, "dashboard", "live_predictions.csv")
COMP_PATH = os.path.join(BASE_DIR, "models", "before_after_comparison.csv")
FPR_PATH = os.path.join(BASE_DIR, "models", "fpr_before_after_comparison.csv")

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

st.set_page_config(
    page_title="Toxic Comment & Fairness Monitor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Sidebar Controls ----------------
st.sidebar.title("⚙️ Stream & Model Settings")
st.sidebar.markdown("---")

source_type = st.sidebar.radio("Live Ingestion Source:", ["YouTube Video", "Reddit Community / Post"])

video_id_input = ""
reddit_target_input = ""

if source_type == "YouTube Video":
    video_id_input = st.sidebar.text_input("YouTube Video ID / URL", value="kffacxfA7G4")
    api_key_input = st.sidebar.text_input("YouTube API Key (optional)", value="", type="password")
else:
    reddit_target_input = st.sidebar.text_input("Subreddit or Post URL", value="r/technology")

st.sidebar.markdown("---")
model_type = st.sidebar.selectbox(
    "Active Scoring Engine:",
    ["🧠 Transformer (Toxic-BERT)", "⚡ TF-IDF Mitigated (BoW)"],
)
st.sidebar.caption("Transformer evaluates semantic context, negation, and benign identity assertions.")

refresh_rate = st.sidebar.slider("Auto-refresh interval (seconds)", min_value=3, max_value=20, value=5)


# ---------------- Cached Data Loaders ----------------
def load_live_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_PATH)
        df["identity_flags"] = df["identity_flags"].fillna("")
        if "source" not in df.columns:
            df["source"] = "youtube"
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_youtube_meta(video_id, api_key):
    if build is None or not video_id or not api_key:
        return None
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return None
        snippet = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        return {
            "title": snippet.get("title"),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle"),
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
        }
    except Exception:
        return None


# ---------------- Header & Main Tabs ----------------
st.title("🎯 Real-Time Toxic Comment Detection — Fairness & Explainability")

tab_stream, tab_lime, tab_bias = st.tabs([
    "🔴 Live Stream Monitor",
    "🔬 LIME Explainability",
    "⚖️ Bias & Fairness Dashboard",
])


# ================= TAB 1: LIVE STREAM MONITOR =================
with tab_stream:
    df_live = load_live_data()

    if df_live.empty:
        st.info("💡 Waiting for incoming live data. Start the background producer or run `python web/app.py`.")
    else:
        left_col, right_col = st.columns([1, 1.2])

        with left_col:
            st.subheader("📺 Source Media Feed")
            if source_type == "YouTube Video":
                active_vid = video_id_input or str(df_live["video_id"].iloc[-1])
                st.video(f"https://www.youtube.com/watch?v={active_vid}")
                meta = fetch_youtube_meta(active_vid, api_key_input)
                if meta:
                    st.write(f"**{meta['title']}**")
                    st.caption(f"Channel: {meta['channel']} | Views: {meta.get('views', 'N/A')} | Likes: {meta.get('likes', 'N/A')}")
            else:
                target_sub = reddit_target_input or "r/technology"
                st.markdown(
                    f"""
                    <div style="background:#1c1824; border:1px solid #ff4500; border-radius:8px; padding:16px; margin-bottom:12px;">
                        <span style="background:#ff4500; color:white; padding:3px 8px; border-radius:12px; font-weight:700; font-size:0.8rem;">REDDIT LIVE</span>
                        <h3 style="margin-top:8px; color:white;">Live Stream: {target_sub}</h3>
                        <p style="color:#a0a7bd; font-size:0.85rem;">Ingesting public Reddit discussion threads in real-time.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.subheader("📊 Live Fairness & Toxicity Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Scored", len(df_live))
            c2.metric("Toxicity Rate", f"{df_live['is_toxic'].mean()*100:.1f}%")
            c3.metric("Identity Mentions", int((df_live["identity_flags"] != "").sum()))

            exploded = df_live.assign(identity_flags=df_live["identity_flags"].str.split(",")).explode("identity_flags")
            exploded = exploded[exploded["identity_flags"] != ""]

            if not exploded.empty:
                st.caption("Toxicity Rate by Identity Mention:")
                group_rates = exploded.groupby("identity_flags")["is_toxic"].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(6, 2.8))
                colors = ["#e15759" if v >= 0.5 else "#59a14f" for v in group_rates.values]
                ax.barh(group_rates.index, group_rates.values * 100, color=colors)
                ax.axvline(df_live["is_toxic"].mean() * 100, color="gray", linestyle="--", linewidth=1, label="Overall Rate")
                ax.set_xlabel("Toxicity Rate (%)")
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("No identity-tagged comments scored yet.")

        with right_col:
            st.subheader("💬 Live Comment Stream")

            filter_mode = st.radio("Filter Feed:", ["All", "🔴 Toxic Only", "🟢 Clean Only"], horizontal=True)

            view_df = df_live.sort_values("scored_at", ascending=False)
            if filter_mode == "🔴 Toxic Only":
                view_df = view_df[view_df["is_toxic"] == True]
            elif filter_mode == "🟢 Clean Only":
                view_df = view_df[view_df["is_toxic"] == False]

            st.caption(f"Displaying newest {min(30, len(view_df))} of {len(view_df)} comments")

            for _, row in view_df.head(30).iterrows():
                is_toxic = row["is_toxic"]
                border_color = "#e15759" if is_toxic else "#59a14f"
                label = "🔴 TOXIC" if is_toxic else "🟢 CLEAN"
                src_badge = f"<span style='font-size:0.7rem; background:#333; padding:2px 5px; border-radius:3px;'>{row.get('source', 'stream').upper()}</span>"
                id_badge = f" &bull; 🏷 {row['identity_flags']}" if row["identity_flags"] else ""

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {border_color}; padding: 10px 14px; margin-bottom: 8px; background-color: rgba(128,128,128,0.06); border-radius: 6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span>{src_badge} <b>{html.escape(str(row['author']))}</b></span>
                            <span style="color:{border_color}; font-weight:700; font-size:0.82rem;">{label} ({row['toxicity_prob']:.2f})</span>
                        </div>
                        <div style="font-size:0.92rem; margin-bottom:4px;">{html.escape(str(row['text']))}</div>
                        <span style="font-size:0.75rem; color:gray;">{row['scored_at']}{id_badge}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ================= TAB 2: LIME EXPLAINABILITY =================
with tab_lime:
    st.subheader("🔬 Local Interpretable Model-agnostic Explanations (LIME)")
    st.markdown(
        "Inspect token-level attribution weights to understand **why** the model made its prediction. "
        "Positive weights (red) increase toxicity probability, while negative weights (green) dampen it."
    )

    explainer = get_explainer()

    col_input, col_config = st.columns([2.5, 1])
    with col_input:
        preset_choice = st.selectbox(
            "Quick Context-Aware Presets:",
            [
                "Custom Text",
                "I am a proud gay woman and I love my community",
                "You are not an idiot, you did an amazing job!",
                "The kill streak in this video game was incredible",
                "You are an absolute idiot and nobody likes you",
                "As a Muslim, peace and tolerance are core to my faith",
            ],
        )

        default_text = "You are not an idiot, you did an amazing job!" if preset_choice == "Custom Text" else preset_choice
        test_comment = st.text_area("Comment to Explain:", value=default_text, height=100)

    with col_config:
        model_choice_lime = st.selectbox("Model to Explain:", ["transformer", "mitigated"], format_func=lambda x: "🧠 Transformer (Context-Aware)" if x == "transformer" else "⚡ TF-IDF Mitigated")
        num_features = st.slider("Number of Word Features:", min_value=4, max_value=12, value=8)
        explain_btn = st.button("🔬 Generate LIME Explanation", use_container_width=True, type="primary")

    if explain_btn or preset_choice != "Custom Text":
        with st.spinner("Generating local linear explanation via text perturbation..."):
            explanation = explainer.explain_comment(
                text=test_comment,
                model_type=model_choice_lime,
                num_features=num_features,
                num_samples=120,
            )

        pred = explanation["prediction"]
        is_toxic = pred["is_toxic"]
        color = "#e15759" if is_toxic else "#59a14f"

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            st.markdown(f"### Prediction: <span style='color:{color}; font-weight:800;'>{pred['label'].upper()}</span>", unsafe_allow_html=True)
            st.metric("Toxicity Probability", f"{pred['toxicity_prob']*100:.1f}%")
            st.metric("Non-Toxicity Probability", f"{pred['non_toxicity_prob']*100:.1f}%")

            st.markdown("#### Highlighted Token Heatmap:")
            st.markdown(
                f"""
                <div style="background-color:rgba(128,128,128,0.1); padding:16px; border-radius:8px; font-size:1.1rem; line-height:1.8;">
                    {explanation['highlighted_html']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            st.markdown("#### Feature Importance Weights:")
            features = explanation["features"]
            if features:
                words = [f["word"] for f in reversed(features)]
                weights = [f["weight"] for f in reversed(features)]
                bar_colors = ["#e15759" if w > 0 else "#59a14f" for w in weights]

                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.barh(words, weights, color=bar_colors)
                ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
                ax.set_xlabel("LIME Weight (Negative = Non-Toxic, Positive = Toxic)")
                plt.tight_layout()
                st.pyplot(fig)

            st.info(
                "💡 **How LIME Works**: LIME randomly perturbs words in the text, queries the classifier, "
                "and fits a local surrogate interpretable linear model to measure each word's contribution."
            )


# ================= TAB 3: BIAS & FAIRNESS DASHBOARD =================
with tab_bias:
    st.subheader("⚖️ Bias & Fairness Benchmark Dashboard")
    st.markdown(
        "Toxicity models trained on unmitigated datasets suffer from unintended bias, generating high false positive rates "
        "when innocent comments contain identity terms (e.g. *gay*, *muslim*, *black*, *woman*). "
        "Below is the benchmark comparing the baseline unmitigated model against our reweighed and post-processed models."
    )

    if os.path.exists(COMP_PATH) and os.path.exists(FPR_PATH):
        df_comp = pd.read_csv(COMP_PATH)
        df_fpr = pd.read_csv(FPR_PATH)
        merged = df_comp.merge(df_fpr, on="identity")

        # Top summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        bpsn_base_mean = df_comp["bpsn_auc_baseline"].mean()
        bpsn_mit_mean = df_comp["bpsn_auc_mitigated"].mean()
        bpsn_gain = (bpsn_mit_mean - bpsn_base_mean) / bpsn_base_mean * 100

        max_fpr_base = df_fpr["fpr_baseline"].max() * 100
        min_fpr_base = df_fpr["fpr_baseline"].min() * 100
        base_disparity = max_fpr_base - min_fpr_base

        max_fpr_post = df_fpr["fpr_postproc"].max() * 100
        min_fpr_post = df_fpr["fpr_postproc"].min() * 100
        post_disparity = max_fpr_post - min_fpr_post

        k1.metric("Mean BPSN AUC Gain", f"+{bpsn_gain:.1f}%", help="Higher BPSN AUC indicates fewer false alarms on identity terms")
        k2.metric("FPR Disparity Reduction", f"{(base_disparity - post_disparity)/base_disparity*100:.1f}%")
        k3.metric("Baseline FPR Disparity Spread", f"{base_disparity:.1f}%")
        k4.metric("Post-Proc Disparity Spread", f"{post_disparity:.1f}%", delta=f"-{base_disparity - post_disparity:.1f}%")

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### BPSN AUC (Baseline vs Mitigated)")
            st.caption("Background Positive, Subgroup Negative: Evaluates false positives on benign identity mentions.")

            fig, ax = plt.subplots(figsize=(6, 4))
            y = np.arange(len(df_comp))
            width = 0.35
            ax.barh(y - width/2, df_comp["bpsn_auc_baseline"], width, label="Baseline", color="#7c7f91")
            ax.barh(y + width/2, df_comp["bpsn_auc_mitigated"], width, label="Mitigated", color="#59a14f")
            ax.set_yticks(y)
            ax.set_yticklabels(df_comp["identity"].str.replace("_", " ").str.title(), fontsize=8)
            ax.set_xlabel("BPSN AUC")
            ax.set_xlim(0.5, 1.0)
            ax.legend(loc="lower right")
            plt.tight_layout()
            st.pyplot(fig)

        with chart_col2:
            st.markdown("#### False Positive Rate (FPR) Parity Comparison (%)")
            st.caption("Reweighing and threshold adjustment flattens the disparity spread across groups.")

            fig, ax = plt.subplots(figsize=(6, 4))
            y = np.arange(len(df_fpr))
            ax.barh(y - width/2, df_fpr["fpr_baseline"] * 100, width, label="Baseline FPR", color="#e15759")
            ax.barh(y + width/2, df_fpr["fpr_postproc"] * 100, width, label="Post-Proc FPR", color="#4a7cff")
            ax.set_yticks(y)
            ax.set_yticklabels(df_fpr["identity"].str.replace("_", " ").str.title(), fontsize=8)
            ax.set_xlabel("FPR (%)")
            ax.legend(loc="upper right")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### Complete Group Fairness Evaluation Table")
        display_df = pd.DataFrame({
            "Identity Group": merged["identity"].str.replace("_", " ").str.title(),
            "Subgroup AUC (Base)": merged["subgroup_auc_baseline"].round(4),
            "Subgroup AUC (Mit)": merged["subgroup_auc_mitigated"].round(4),
            "BPSN AUC (Base)": merged["bpsn_auc_baseline"].round(4),
            "BPSN AUC (Mit)": merged["bpsn_auc_mitigated"].round(4),
            "FPR Baseline (%)": (merged["fpr_baseline"] * 100).round(1),
            "FPR Mitigated (%)": (merged["fpr_mitigated"] * 100).round(1),
            "FPR Post-Proc (%)": (merged["fpr_postproc"] * 100).round(1),
        })
        st.dataframe(display_df, use_container_width=True)

    else:
        st.warning("Fairness comparison files not found in models/ directory. Run `python src/bias_mitigation.py` first.")