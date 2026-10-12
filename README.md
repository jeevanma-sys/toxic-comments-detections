# 🎯 Real-Time Toxic Comment Detection & Fairness Monitor

A comprehensive AI pipeline for real-time toxicity detection, ethical bias mitigation, token-level interpretability (LIME), and multi-platform live streaming (YouTube & Reddit).

---

## 🌟 Key Features

### 1. 🧠 Transformer-Based Context-Aware Scoring
- **Deep Contextual Understanding:** Powered by `unitary/toxic-bert` (BERT fine-tuned on Jigsaw Toxicity data) with fallback to TF-IDF mitigated models.
- **Solves Bag-of-Words (TF-IDF) Failures:**
  - **Negation Handling:** Accurately recognizes context such as *"You are not an idiot, you did great!"* (TF-IDF flags "idiot" as high toxicity, while Transformer evaluates the sentence semantics).
  - **Benign Identity Reclamation:** Differentiates between benign identity assertions (*"I am a proud gay woman"*, *"As a Muslim, I love peace"*) and identity-targeted hate speech.
  - **Vernacular & Gaming Slang:** Distinguishes benign expressions (*"The kill streak in this game was epic"*) from real threats.
- **Multi-Class Detection:** Detects `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

### 2. 🔬 LIME Explainability Panel & Bias Dashboard
- **Local Interpretable Model-agnostic Explanations (LIME):**
  - Perturbs inputs to calculate per-token attribution weights.
  - Renders inline token heatmaps (green for calming/non-toxic tokens, red for toxic trigger words).
  - Feature importance bar chart visualizing exact word attribution values.
  - Model comparison mode: contrast how the Transformer reasons compared to traditional BoW models.
- **⚖️ Bias & Fairness Dashboard:**
  - Quantifies unintended demographic bias across 9 protected identity groups (`female`, `male`, `homosexual_gay_or_lesbian`, `muslim`, `christian`, `jewish`, `black`, `white`, `psychiatric_or_mental_illness`).
  - **BPSN AUC (Background Positive, Subgroup Negative):** Tracks reduction in false alarms on innocent identity mentions.
  - **FPR Parity & Equalized Odds:** Demonstrates how sample reweighing and threshold post-processing reduce false positive rate disparities.

### 3. 🤖 Optional Reddit API as Second Live Source
- **Dual Platform Ingestion:** Monitor comments in real-time from either **YouTube** videos or **Reddit** communities/posts.
- **Zero-Setup Mode:** Ingests live comments directly from public Reddit feeds (`r/technology`, `r/AskReddit`, `r/all`, etc.) without requiring developer API keys!
- **Authenticated OAuth Mode:** Supports Reddit Application Credentials (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`) for high-throughput streaming.

---

## 📁 Repository Structure

```
Toxic-Comment-main/
│
├── dashboard/
│   ├── app.py                      # Multi-tab Streamlit dashboard (Live Stream, LIME, Bias)
│   └── live_predictions.csv       # Live prediction store
│
├── src/
│   ├── transformer_scorer.py       # Context-aware Transformer scoring & predict_proba
│   ├── explainability.py           # LIME explainer with token heatmap generation
│   ├── reddit_source.py            # Reddit live ingestion engine (RSS + OAuth)
│   ├── reddit_producer.py          # Kafka producer for Reddit comments
│   ├── realtime_scorer.py          # Unified CLI scorer for YouTube & Reddit
│   ├── kafka_producer.py           # Kafka producer for YouTube comments
│   ├── kafka_consumer.py           # Kafka consumer with Transformer scoring
│   ├── bias_mitigation.py          # Sample reweighing & threshold post-processing
│   └── fairness_utils.py           # Subgroup, BPSN, BNSP AUC evaluation logic
│
├── web/
│   ├── app.py                      # Flask backend API (/api/start, /api/data, /api/explain, /api/bias_metrics)
│   ├── templates/
│   │   └── index.html              # Multi-tab web UI (Live Monitor, LIME Panel, Bias Dashboard)
│   └── static/
│       ├── script.js               # Frontend controller & real-time poller
│       └── style.css               # Modern dark-theme UI styles
│
├── models/
│   ├── before_after_comparison.csv     # Pre-computed BPSN / BNSP AUC benchmark
│   └── fpr_before_after_comparison.csv # False positive rate disparity comparison
│
├── tests/
│   └── test_e2e_features.py       # Comprehensive automated test suite
├── docker-compose.yml              # Kafka + Zookeeper infrastructure
└── requirements.txt                # Python dependencies
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Web Application (Flask)
Start the complete web portal featuring the Live Stream Monitor, LIME Explainability Panel, and Bias Dashboard:
```bash
python web/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.
- **YouTube Live:** Paste any YouTube link (e.g. `https://www.youtube.com/watch?v=kffacxfA7G4`) and click **Start Stream**.
- **Reddit Live:** Toggle the source button to **Reddit**, enter any subreddit (e.g. `r/technology` or `r/AskReddit`), and watch live comments streamed and scored!
- **LIME Explainability:** Click **"🔬 Explain with LIME"** on any live comment or switch to the **LIME Explainability** tab to test custom sentences with interactive heatmaps.
- **Bias & Fairness Dashboard:** Switch to the **Bias & Fairness** tab to inspect demographic parity, BPSN AUC gains, and FPR disparity reduction tables.

---

### 3. Run the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Includes:
- **Tab 1:** Live Monitor with video/Reddit embedding and comment feed filtering.
- **Tab 2:** Interactive LIME token attribution playground with Matplotlib weight charts.
- **Tab 3:** Side-by-side grouped bar charts for BPSN AUC and FPR disparity parity.

---

### 4. CLI & Standalone Tools

#### Benchmark Transformer Context Awareness
```bash
python src/transformer_scorer.py --test
```

#### Run LIME Explanation from Terminal
```bash
python src/explainability.py --text "You are not an idiot, you did an amazing job!" --model transformer
```

#### Stream Live Reddit Comments from Terminal
```bash
python src/realtime_scorer.py --source reddit --target "technology" --model transformer
```

#### Run Automated Test Suite
```bash
python tests/test_e2e_features.py
```

---

## ⚖️ Fairness & Bias Mitigation Metrics

| Metric | Baseline | Mitigated (Reweighing + Post-Processing) | Impact |
| :--- | :---: | :---: | :--- |
| **Mean BPSN AUC** | 0.7410 | **0.7570** (+2.1%) | Fewer false alarms when innocent comments contain identity keywords. |
| **FPR Disparity Spread** | 45.4% | **19.4%** (57.3% reduction) | Equalized False Positive Rates across demographic groups. |
| **Context False Positives** | High in BoW | **Near Zero in Transformer** | "I am a proud gay woman" drops from ~36% BoW toxicity to **0.95%** in Transformer. |
