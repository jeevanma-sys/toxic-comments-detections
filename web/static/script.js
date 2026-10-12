let currentFilter = "all";
let currentTargetSource = "youtube";
let currentTargetId = null;
let biasMetricsLoaded = false;

// ================= TAB SWITCHING =================
document.querySelectorAll(".nav-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    tab.classList.add("active");
    const targetId = tab.dataset.tab;
    const content = document.getElementById(targetId);
    if (content) content.classList.add("active");

    if (targetId === "tab-bias" && !biasMetricsLoaded) {
      loadBiasDashboard();
    }
  });
});

// ================= SOURCE SELECTION =================
const srcYtBtn = document.getElementById("srcYtBtn");
const srcRedditBtn = document.getElementById("srcRedditBtn");
const targetInput = document.getElementById("targetInput");
const youtubeContainer = document.getElementById("youtubeContainer");
const redditContainer = document.getElementById("redditContainer");

if (srcYtBtn && srcRedditBtn) {
  srcYtBtn.addEventListener("click", () => {
    currentTargetSource = "youtube";
    srcYtBtn.classList.add("active");
    srcRedditBtn.classList.remove("active");
    targetInput.placeholder = "Paste any YouTube video URL or ID (e.g. https://www.youtube.com/watch?v=kffacxfA7G4)...";
    youtubeContainer.style.display = "block";
    redditContainer.style.display = "none";
  });

  srcRedditBtn.addEventListener("click", () => {
    currentTargetSource = "reddit";
    srcRedditBtn.classList.add("active");
    srcYtBtn.classList.remove("active");
    targetInput.placeholder = "Enter Subreddit (e.g. r/technology, r/AskReddit, r/worldnews) or Reddit post link...";
    youtubeContainer.style.display = "none";
    redditContainer.style.display = "block";
  });
}

// ================= START STREAM =================
document.getElementById("goBtn").addEventListener("click", async () => {
  const url = targetInput.value.trim();
  const statusEl = document.getElementById("statusText");
  const modelChoice = document.getElementById("modelSelect").value;

  if (!url) {
    statusEl.textContent = "Please enter a URL or target name.";
    statusEl.className = "status-text error";
    return;
  }

  statusEl.textContent = "Connecting to source...";
  statusEl.className = "status-text";

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: currentTargetSource,
        url: url,
        model: modelChoice,
      }),
    });
    const result = await res.json();

    if (result.ok) {
      currentTargetId = result.target;
      loadSourceMeta(result.source, result.target);
      statusEl.textContent = `● Live Stream Active (${result.source.toUpperCase()})`;
      statusEl.className = "status-text ok";
      document.getElementById("connectionPill").textContent = "● Live";
      document.getElementById("connectionPill").className = "pill pill-status";
      
      // Initial prompt
      const feed = document.getElementById("commentFeed");
      feed.innerHTML = '<p class="empty-state">● Ingesting & scoring comments with Toxic-BERT...</p>';
      
      setTimeout(refreshData, 1200);
      setTimeout(refreshData, 3000);
    } else {
      statusEl.textContent = "⚠ " + (result.error || "Failed to start");
      statusEl.className = "status-text error";
    }
  } catch (err) {
    statusEl.textContent = "⚠ Connection error to server";
    statusEl.className = "status-text error";
  }
});

// ================= FILTER BUTTONS =================
document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    refreshData();
  });
});

// ================= LIVE STREAM DATA POLLING =================
async function refreshData() {
  try {
    const res = await fetch(`/api/data?filter=${currentFilter}`);
    const data = await res.json();

    const statusEl = document.getElementById("statusText");

    if (data.error) {
      console.warn("API Data warning:", data.error);
      if (statusEl && !statusEl.textContent.includes("Active")) {
        statusEl.textContent = "⚠ " + data.error;
        statusEl.className = "status-text error";
      }
      return;
    }

    if (data.status === "error" && data.last_error) {
      statusEl.textContent = "⚠ " + data.last_error;
      statusEl.className = "status-text error";
    } else if (data.status === "streaming" && !statusEl.textContent.includes("error")) {
      statusEl.textContent = `● Live (${(data.source || currentTargetSource).toUpperCase()})`;
      statusEl.className = "status-text ok";
    }

    document.getElementById("totalCount").textContent = data.total;
    document.getElementById("toxicityRate").textContent = data.toxicity_rate + "%";
    document.getElementById("identityCount").textContent = data.identity_count;
    document.getElementById("commentCount").textContent = `${data.comments.length} comments`;

    // Update active model pill
    const modelPill = document.getElementById("modelPill");
    if (data.model === "transformer") {
      modelPill.textContent = "🧠 Toxic-BERT";
    } else {
      modelPill.textContent = "⚡ TF-IDF Mitigated";
    }

    // Handle target media change if updated on server
    if (data.target && data.target !== currentTargetId) {
      currentTargetId = data.target;
      loadSourceMeta(data.source, data.target);
    }

    renderIdentityChart(data.identity_rates || []);
    renderComments(data.comments || []);
  } catch (err) {
    console.error("Live poll failed:", err);
  }
}

// ================= RENDER MEDIA & SOURCE METADATA =================
async function loadSourceMeta(source, target) {
  if (source === "reddit") {
    youtubeContainer.style.display = "none";
    redditContainer.style.display = "block";
    document.getElementById("redditSub").textContent = target;
    document.getElementById("redditTitle").textContent = `Live Feed: ${target}`;
    document.getElementById("redditAuthor").textContent = `Source: Reddit Community`;
    document.getElementById("redditLink").href = target.startsWith("http") ? target : `https://reddit.com/r/${target}`;
  } else {
    youtubeContainer.style.display = "block";
    redditContainer.style.display = "none";
    const ytFrame = document.getElementById("ytFrame");
    if (ytFrame) {
      ytFrame.src = `https://www.youtube.com/embed/${target}`;
    }
    const watchLink = document.getElementById("watchLink");
    const watchHref = document.getElementById("watchLinkHref");
    watchHref.href = `https://www.youtube.com/watch?v=${target}`;
    watchLink.style.display = "block";
  }

  try {
    const res = await fetch("/api/source_meta");
    const meta = await res.json();

    if (source === "reddit" && meta.title) {
      document.getElementById("redditTitle").textContent = meta.title;
      document.getElementById("redditAuthor").textContent = `Source: ${meta.author || 'Reddit Community'}`;
    } else if (source === "youtube") {
      const card = document.getElementById("sourceInfoCard");
      if (meta.title) {
        document.getElementById("sourceTitle").textContent = meta.title;
        document.getElementById("sourceSub").textContent =
          `${meta.channel || ""} · ${meta.views || "?"} views · ${meta.likes || "?"} likes`;
        card.style.display = "block";
      }
    }
  } catch (e) {
    console.error("Failed loading source meta:", e);
  }
}

// ================= RENDER LIVE COMMENTS =================
function renderComments(comments) {
  const feed = document.getElementById("commentFeed");

  if (comments.length === 0) {
    if (!feed.innerHTML.includes("Ingesting")) {
      feed.innerHTML = '<p class="empty-state">No comments yet — waiting for incoming stream data...</p>';
    }
    return;
  }

  feed.innerHTML = "";
  comments.forEach(c => {
    const card = document.createElement("div");
    card.className = "comment-card" + (c.is_toxic ? " toxic" : "");

    const label = c.is_toxic ? "🔴 TOXIC" : "🟢 CLEAN";
    const labelClass = c.is_toxic ? "toxic" : "clean";
    const identityBadge = c.identity_flags ? ` · 🏷 ${c.identity_flags}` : "";

    const isReddit = (c.source === "reddit" || (c.comment_id && c.comment_id.startsWith("rc_")));
    const sourceTag = isReddit
      ? '<span class="source-tag reddit">Reddit</span>'
      : '<span class="source-tag yt">YouTube</span>';

    card.innerHTML = `
      <div class="comment-top">
        <div class="comment-author-wrap">
          ${sourceTag}
          <span class="comment-author">${escapeHtml(c.author || "Unknown")}</span>
        </div>
        <span class="comment-label ${labelClass}">${label} (${c.toxicity_prob.toFixed(2)})</span>
      </div>
      <div class="comment-text">${escapeHtml(c.text || "")}</div>
      <div class="comment-bottom">
        <span class="comment-meta">${escapeHtml(c.scored_at || "")}${identityBadge}</span>
        <button class="explain-btn" data-text="${escapeAttr(c.text || '')}">🔬 Explain with LIME</button>
      </div>
    `;

    // Attach LIME explanation click
    const expBtn = card.querySelector(".explain-btn");
    expBtn.addEventListener("click", () => {
      openQuickLimeModal(c.text);
    });

    feed.appendChild(card);
  });
}

function renderIdentityChart(rates) {
  const container = document.getElementById("identityChart");
  container.innerHTML = "";

  if (rates.length === 0) {
    container.innerHTML = '<p class="empty-state">No identity-tagged comments yet.</p>';
    return;
  }

  rates.forEach(item => {
    const row = document.createElement("div");
    row.className = "identity-row";
    row.innerHTML = `
      <span class="identity-name" title="${item.identity}">${item.identity}</span>
      <div class="identity-bar-bg">
        <div class="identity-bar-fill" style="width:${Math.min(100, item.rate)}%"></div>
      </div>
      <span class="identity-pct">${item.rate}%</span>
    `;
    container.appendChild(row);
  });
}

// ================= TAB 2: LIME EXPLAINABILITY =================
const limeTextInput = document.getElementById("limeTextInput");
const limeModelSelect = document.getElementById("limeModelSelect");
const runLimeBtn = document.getElementById("runLimeBtn");
const limePlaceholder = document.getElementById("limePlaceholder");
const limeOutput = document.getElementById("limeOutput");

// Preset chips click
document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    limeTextInput.value = btn.dataset.text;
    runLimeAnalysis(btn.dataset.text, limeModelSelect.value);
  });
});

if (runLimeBtn) {
  runLimeBtn.addEventListener("click", () => {
    const text = limeTextInput.value.trim();
    if (!text) {
      alert("Please enter a sentence to explain.");
      return;
    }
    runLimeAnalysis(text, limeModelSelect.value);
  });
}

async function runLimeAnalysis(text, model) {
  limePlaceholder.style.display = "block";
  limePlaceholder.innerHTML = '<div class="spinner"></div><p style="text-align:center; color:#a0a7bd;">Perturbing text & running LIME explanation...</p>';
  limeOutput.style.display = "none";

  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    const data = await res.json();

    if (data.error) {
      limePlaceholder.innerHTML = `<p class="empty-state" style="color:#e15759;">Error: ${data.error}</p>`;
      return;
    }

    renderLimeResults(data);
    limePlaceholder.style.display = "none";
    limeOutput.style.display = "block";
  } catch (err) {
    limePlaceholder.innerHTML = `<p class="empty-state" style="color:#e15759;">Failed to generate LIME explanation.</p>`;
  }
}

function renderLimeResults(data) {
  const pred = data.prediction;
  const isToxic = pred.is_toxic;

  // Prediction Banner
  const predLabel = document.getElementById("predLabel");
  const predProb = document.getElementById("predProb");
  const probBarFill = document.getElementById("probBarFill");

  predLabel.textContent = isToxic ? "🔴 TOXIC" : "🟢 NON-TOXIC";
  predLabel.className = "pred-label " + (isToxic ? "toxic" : "clean");
  predProb.textContent = `${(pred.toxicity_prob * 100).toFixed(1)}% Toxicity Probability`;
  probBarFill.style.width = `${Math.min(100, Math.max(0, pred.toxicity_prob * 100))}%`;

  // Token Heatmap
  document.getElementById("tokenHeatmap").innerHTML = data.highlighted_html || escapeHtml(data.text);

  // Feature Importance Chart
  const featureChart = document.getElementById("featureChart");
  featureChart.innerHTML = "";

  const maxAbs = Math.max(...data.features.map(f => f.abs_weight), 0.01);

  data.features.forEach(f => {
    const row = document.createElement("div");
    row.className = "feature-row";
    const widthPct = Math.min(100, (f.abs_weight / maxAbs) * 100);
    const directionClass = f.direction === "toxic" ? "toxic" : "clean";
    const sign = f.weight > 0 ? "+" : "";

    row.innerHTML = `
      <span class="feature-word" title="${f.word}">${f.word}</span>
      <div class="feature-bar-wrap">
        <div class="feature-bar-inner ${directionClass}" style="width:${widthPct}%"></div>
      </div>
      <span class="feature-weight" style="color:${f.direction === 'toxic' ? '#e15759' : '#59a14f'};">
        ${sign}${f.weight.toFixed(3)}
      </span>
    `;
    featureChart.appendChild(row);
  });

  // Context commentary
  const insightText = document.getElementById("contextInsightText");
  const textLower = data.text.toLowerCase();
  let insight = "";

  if (textLower.includes("proud gay") || textLower.includes("muslim") || textLower.includes("woman") || textLower.includes("christian")) {
    if (!isToxic) {
      insight = "✅ Context-Aware Benefit: The model correctly identified benign/positive identity terms, avoiding false positive penalties that naive bag-of-words models frequently assign to protected group keywords.";
    } else {
      insight = "⚠️ Identity-targeted risk detected: The comment combines sensitive identity terms with aggressive or derogatory framing.";
    }
  } else if (textLower.includes("not ") || textLower.includes("n't ")) {
    insight = "💡 Negation Understanding: The transformer contextualizes negation words (e.g. 'not', 'never'), dynamically dampening toxicity that would otherwise trigger on standalone negative words.";
  } else if (isToxic) {
    insight = "⚠️ Severe hostility or derogatory terms dominate the local feature attribution weights.";
  } else {
    insight = "🟢 Non-toxic framing: Neutral discourse or constructive sentiment detected across token interactions.";
  }
  insightText.textContent = insight;
}

// ================= QUICK LIME MODAL =================
const limeModal = document.getElementById("limeModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalBody = document.getElementById("modalBody");

if (closeModalBtn) {
  closeModalBtn.addEventListener("click", () => {
    limeModal.style.display = "none";
  });
}
window.addEventListener("click", (e) => {
  if (e.target === limeModal) limeModal.style.display = "none";
});

async function openQuickLimeModal(text) {
  limeModal.style.display = "flex";
  modalBody.innerHTML = '<div class="spinner"></div><p style="text-align:center; color:#a0a7bd;">Computing LIME attribution...</p>';

  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: "transformer" }),
    });
    const data = await res.json();

    if (data.error) {
      modalBody.innerHTML = `<p style="color:#e15759;">Error: ${data.error}</p>`;
      return;
    }

    const isToxic = data.prediction.is_toxic;
    const maxAbs = Math.max(...data.features.map(f => f.abs_weight), 0.01);

    let featuresHtml = data.features.map(f => {
      const widthPct = Math.min(100, (f.abs_weight / maxAbs) * 100);
      const color = f.direction === 'toxic' ? '#e15759' : '#59a14f';
      const sign = f.weight > 0 ? "+" : "";
      return `
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px; font-size:0.82rem;">
          <span style="width:110px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:600;">${f.word}</span>
          <div style="flex:1; height:12px; background:#1e2338; border-radius:3px; overflow:hidden;">
            <div style="height:100%; width:${widthPct}%; background:${color}; border-radius:3px;"></div>
          </div>
          <span style="width:55px; text-align:right; font-weight:700; color:${color};">${sign}${f.weight.toFixed(3)}</span>
        </div>
      `;
    }).join("");

    modalBody.innerHTML = `
      <div style="margin-bottom:14px; padding:10px; background:#1b1f32; border-radius:6px;">
        <span style="font-weight:700; font-size:1.1rem; color:${isToxic ? '#e15759' : '#59a14f'}">
          ${isToxic ? '🔴 TOXIC' : '🟢 NON-TOXIC'}
        </span>
        <span style="float:right; font-size:0.85rem; color:#a0a7bd;">
          ${(data.prediction.toxicity_prob * 100).toFixed(1)}% prob
        </span>
      </div>

      <div style="margin-bottom:16px;">
        <h4 style="font-size:0.9rem; margin-bottom:6px; color:#ffffff;">Token Impact Heatmap:</h4>
        <div style="background:#161927; padding:12px; border-radius:6px; border:1px solid #272d42; line-height:1.7;">
          ${data.highlighted_html}
        </div>
      </div>

      <div>
        <h4 style="font-size:0.9rem; margin-bottom:6px; color:#ffffff;">Word Attribution Weights:</h4>
        ${featuresHtml}
      </div>

      <div style="text-align:right; margin-top:14px;">
        <button id="jumpToLimeBtn" class="primary-btn" style="padding:6px 14px; font-size:0.8rem;">
          Open in Full LIME Playground →
        </button>
      </div>
    `;

    document.getElementById("jumpToLimeBtn").addEventListener("click", () => {
      limeModal.style.display = "none";
      document.querySelector('[data-tab="tab-lime"]').click();
      limeTextInput.value = text;
      runLimeAnalysis(text, "transformer");
    });

  } catch (err) {
    modalBody.innerHTML = '<p style="color:#e15759;">Failed to generate LIME explanation.</p>';
  }
}

// ================= TAB 3: BIAS & FAIRNESS DASHBOARD =================
async function loadBiasDashboard() {
  try {
    const res = await fetch("/api/bias_metrics");
    const data = await res.json();

    if (!data.identities || data.identities.length === 0) return;
    biasMetricsLoaded = true;

    // 1. KPI cards
    const overall = data.overall;
    if (overall.bpsn_gain_pct) {
      document.getElementById("kpiBpsnGain").textContent = `+${overall.bpsn_gain_pct}%`;
      document.getElementById("kpiDisparityRed").textContent = `${overall.disparity_reduction_pct}%`;
      document.getElementById("kpiBaseDisparity").textContent = `${overall.fpr_disparity_baseline}%`;
      document.getElementById("kpiPostDisparity").textContent = `${overall.fpr_disparity_postproc}%`;
    }

    // 2. BPSN Chart
    const bpsnChart = document.getElementById("bpsnChart");
    bpsnChart.innerHTML = "";

    data.identities.forEach(item => {
      const group = document.createElement("div");
      group.className = "chart-bar-group";

      const basePct = Math.round(item.bpsn_auc_baseline * 100);
      const mitPct = Math.round(item.bpsn_auc_mitigated * 100);

      group.innerHTML = `
        <span class="chart-group-label">${formatIdentityName(item.identity)}</span>
        <div class="chart-bars-wrap">
          <div class="chart-bar-row">
            <span class="bar-tag">Baseline</span>
            <div class="bar-fill-bg">
              <div class="bar-fill" style="width:${basePct}%; background:#6c757d;"></div>
            </div>
            <span class="bar-val">${item.bpsn_auc_baseline.toFixed(3)}</span>
          </div>
          <div class="chart-bar-row">
            <span class="bar-tag" style="color:#59a14f; font-weight:600;">Mitigated</span>
            <div class="bar-fill-bg">
              <div class="bar-fill" style="width:${mitPct}%; background:#59a14f;"></div>
            </div>
            <span class="bar-val" style="color:#59a14f;">${item.bpsn_auc_mitigated.toFixed(3)}</span>
          </div>
        </div>
      `;
      bpsnChart.appendChild(group);
    });

    // 3. FPR Comparison Chart
    const fprChart = document.getElementById("fprChart");
    fprChart.innerHTML = "";

    data.identities.forEach(item => {
      const group = document.createElement("div");
      group.className = "chart-bar-group";

      group.innerHTML = `
        <span class="chart-group-label">${formatIdentityName(item.identity)}</span>
        <div class="chart-bars-wrap">
          <div class="chart-bar-row">
            <span class="bar-tag">Baseline</span>
            <div class="bar-fill-bg">
              <div class="bar-fill" style="width:${Math.min(100, item.fpr_baseline)}%; background:#e15759;"></div>
            </div>
            <span class="bar-val">${item.fpr_baseline}%</span>
          </div>
          <div class="chart-bar-row">
            <span class="bar-tag" style="color:#7aa2f7;">Post-Proc</span>
            <div class="bar-fill-bg">
              <div class="bar-fill" style="width:${Math.min(100, item.fpr_postproc)}%; background:#7aa2f7;"></div>
            </div>
            <span class="bar-val" style="color:#7aa2f7;">${item.fpr_postproc}%</span>
          </div>
        </div>
      `;
      fprChart.appendChild(group);
    });

    // 4. Tabular Breakdown
    const tbody = document.getElementById("fairnessTableBody");
    tbody.innerHTML = "";

    data.identities.forEach(item => {
      const tr = document.createElement("tr");
      const fprDelta = (item.fpr_postproc - item.fpr_baseline).toFixed(1);
      const deltaColor = item.fpr_postproc <= item.fpr_baseline ? "#59a14f" : "#e15759";

      tr.innerHTML = `
        <td><strong>${formatIdentityName(item.identity)}</strong></td>
        <td>${item.subgroup_auc_baseline}</td>
        <td style="color:#59a14f; font-weight:600;">${item.subgroup_auc_mitigated}</td>
        <td>${item.bpsn_auc_baseline}</td>
        <td style="color:#59a14f; font-weight:600;">${item.bpsn_auc_mitigated}</td>
        <td>${item.fpr_baseline}%</td>
        <td>${item.fpr_mitigated}%</td>
        <td style="color:#7aa2f7; font-weight:600;">${item.fpr_postproc}%</td>
        <td style="color:${deltaColor}; font-weight:600;">${fprDelta > 0 ? '+' : ''}${fprDelta}%</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Failed to load bias metrics:", err);
  }
}

function formatIdentityName(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function escapeAttr(str) {
  return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Initial polling loop
setInterval(refreshData, 3500);
refreshData();