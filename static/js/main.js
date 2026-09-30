/**
 * EchoCheck - AI-Based Fake News Detection System
 * Frontend Application Logic, Hybrid Client/Server ML Engine, and Reactive UI
 */

document.addEventListener('DOMContentLoaded', () => {
  initMobileNav();
  initNewsDetector();
  initHistoryView();
});

/* ==========================================================================
   Mobile Navigation Toggle
   ========================================================================== */
function initMobileNav() {
  const toggleBtn = document.getElementById('mobileNavToggle');
  const navLinks = document.getElementById('navLinks');

  if (toggleBtn && navLinks) {
    toggleBtn.addEventListener('click', () => {
      navLinks.classList.toggle('active');
      const isExpanded = navLinks.classList.contains('active');
      toggleBtn.setAttribute('aria-expanded', isExpanded);
    });
  }
}

/* ==========================================================================
   Client-Side ML Engine (Runs 100% on GitHub Pages without a Python server!)
   ========================================================================== */
let cachedWeights = null;

async function loadModelWeights() {
  if (cachedWeights) return cachedWeights;
  try {
    const res = await fetch('static/model/model_weights.json');
    if (!res.ok) throw new Error('Weights not found');
    cachedWeights = await res.json();
    return cachedWeights;
  } catch (e) {
    console.warn('Could not load static weights file:', e);
    return null;
  }
}

function cleanTextJS(text) {
  let cleaned = (text || '').toLowerCase();
  cleaned = cleaned.replace(/^\s*([a-za-z\s,]+)?\((?:reuters|ap|afp|bloomberg|pti|al jazeera)\)\s*[-–—:]\s*/i, ' ');
  cleaned = cleaned.replace(/\b(reuters|associated press)\b/gi, ' ');
  cleaned = cleaned.replace(/https?:\/\/\S+|www\.\S+/gi, ' ');
  cleaned = cleaned.replace(/<.*?>/g, ' ');
  cleaned = cleaned.replace(/[^a-zA-Z\s]/g, ' ');
  return cleaned.replace(/\s+/g, ' ').trim();
}

function extractLinguisticSignalsJS(rawText) {
  const words = (rawText || '').trim().split(/\s+/).filter(Boolean);
  const totalChars = rawText.length;
  let upper = 0, excl = 0, quest = 0;
  for (let c of rawText) {
    if (c >= 'A' && c <= 'Z') upper++;
    if (c === '!') excl++;
    if (c === '?') quest++;
  }
  const avgLen = words.length > 0 ? (words.reduce((a, b) => a + b.length, 0) / words.length) : 0;
  const upperRatio = totalChars > 0 ? ((upper / totalChars) * 100) : 0;
  return {
    word_count: words.length,
    char_count: totalChars,
    uppercase_ratio: parseFloat(upperRatio.toFixed(2)),
    exclamation_count: excl,
    question_count: quest,
    avg_word_length: parseFloat(avgLen.toFixed(2))
  };
}

async function predictClientSide(rawText) {
  const modelData = await loadModelWeights();
  if (!modelData) {
    throw new Error('Model weights could not be loaded.');
  }

  const cleaned = cleanTextJS(rawText);
  const words = cleaned.split(/\s+/).filter(Boolean);
  const ngrams = {};
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    ngrams[w] = (ngrams[w] || 0) + 1;
    if (i + 1 < words.length) {
      const bg = w + ' ' + words[i + 1];
      ngrams[bg] = (ngrams[bg] || 0) + 1;
    }
  }

  const weights = modelData.weights;
  const intercepts = modelData.intercepts;
  const calibs = modelData.calibrations;

  const vec = {};
  let normSq = 0;
  for (let term in ngrams) {
    if (weights[term]) {
      const idf = weights[term][0];
      const count = ngrams[term];
      const tf = 1.0 + Math.log(count);
      const tfidf = tf * idf;
      vec[term] = tfidf;
      normSq += tfidf * tfidf;
    }
  }

  const norm = Math.sqrt(normSq) || 1.0;
  for (let term in vec) {
    vec[term] /= norm;
  }

  const pRealList = [];
  for (let cIdx = 0; cIdx < 3; cIdx++) {
    let dot = intercepts[cIdx];
    for (let term in vec) {
      dot += vec[term] * weights[term][1 + cIdx];
    }
    const a = calibs[cIdx].a;
    const b = calibs[cIdx].b;
    const pReal = 1.0 / (1.0 + Math.exp(a * dot + b));
    pRealList.push(pReal);
  }

  const realProb = parseFloat(((pRealList.reduce((a, b) => a + b, 0) / 3.0) * 100).toFixed(2));
  const fakeProb = parseFloat((100.0 - realProb).toFixed(2));
  const label = realProb >= 50.0 ? 'REAL' : 'FAKE';
  const confidence = label === 'REAL' ? realProb : fakeProb;

  const signals = extractLinguisticSignalsJS(rawText);

  let explanation = '';
  if (label === 'REAL') {
    if (confidence > 85) {
      explanation = `The article exhibits formal journalistic conventions, objective vocabulary, and balanced syntax characteristic of verified news reporting. Calibrated model confidence is strong (${confidence}%), and linguistic analysis indicates low sensationalism with an uppercase ratio of only ${signals.uppercase_ratio}%.`;
    } else {
      explanation = `The model classifies this text as REAL with moderate confidence (${confidence}%). While the article maintains factual structure and measured vocabulary, some phrasing exhibits stylistic overlap with opinion or commentary pieces.`;
    }
  } else {
    if (signals.uppercase_ratio > 4.0 || signals.exclamation_count > 1) {
      explanation = `The article displays prominent hallmarks of fabricated or deceptive news, including sensationalist phrasing, emotional urgency, and elevated punctuation/capitalization (${signals.exclamation_count} exclamation marks, ${signals.uppercase_ratio}% uppercase). TF-IDF n-grams matched high-frequency lexical patterns found in the disinformation corpus with ${confidence}% calibrated confidence.`;
    } else {
      explanation = `The article is predicted to be FAKE with ${confidence}% confidence. The semantic structure, unverifiable claims, and vocabulary distribution match patterns identified in the disinformation training dataset, diverging significantly from verified journalistic source standards.`;
    }
  }

  return {
    success: true,
    label: label,
    confidence: confidence,
    real_prob: realProb,
    fake_prob: fakeProb,
    signals: signals,
    explanation: explanation,
    timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19)
  };
}

function saveToLocalHistory(item) {
  try {
    const list = JSON.parse(localStorage.getItem('echocheck_history') || '[]');
    list.unshift({
      id: list.length + 1,
      article_text: item.text,
      article_snippet: item.text.replace(/\s+/g, ' ').substring(0, 160) + '...',
      predicted_label: item.label,
      confidence_score: item.confidence,
      real_prob: item.real_prob,
      fake_prob: item.fake_prob,
      created_at: item.timestamp || new Date().toISOString().replace('T', ' ').substring(0, 19)
    });
    localStorage.setItem('echocheck_history', JSON.stringify(list.slice(0, 100)));
  } catch (e) {
    console.warn('LocalStorage save error:', e);
  }
}

/* ==========================================================================
   News Detector Workspace Logic
   ========================================================================== */
function initNewsDetector() {
  const textarea = document.getElementById('newsArticleInput');
  const charCounter = document.getElementById('charCount');
  const wordCounter = document.getElementById('wordCount');
  const clearBtn = document.getElementById('clearTextBtn');
  const analyzeBtn = document.getElementById('analyzeNewsBtn');
  const validationAlert = document.getElementById('validationAlert');
  const alertMessage = document.getElementById('alertMessage');
  const loadingBox = document.getElementById('loadingBox');
  const loadingStatusText = document.getElementById('loadingStatusText');
  const resultsCard = document.getElementById('resultsCard');
  const sampleButtons = document.querySelectorAll('.btn-sample');

  if (!textarea) return; // Not on detect page

  // 1. Textarea Counters
  const updateCounters = () => {
    const text = textarea.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;

    if (charCounter) charCounter.textContent = `${chars.toLocaleString()} characters`;
    if (wordCounter) wordCounter.textContent = `${words.toLocaleString()} words`;

    if (validationAlert && validationAlert.style.display !== 'none') {
      validationAlert.style.display = 'none';
    }
  };

  textarea.addEventListener('input', updateCounters);
  textarea.addEventListener('paste', () => setTimeout(updateCounters, 50));

  // 2. Clear Button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      textarea.value = '';
      updateCounters();
      if (validationAlert) validationAlert.style.display = 'none';
      if (resultsCard) resultsCard.style.display = 'none';
      textarea.focus();
    });
  }

  // 3. Sample Articles Loader
  sampleButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const sampleText = e.currentTarget.getAttribute('data-sample-text');
      if (sampleText) {
        textarea.value = sampleText;
        updateCounters();
        if (validationAlert) validationAlert.style.display = 'none';
        if (resultsCard) resultsCard.style.display = 'none';
        textarea.focus();
      }
    });
  });

  // 4. Form Submission & Hybrid Real-Time Prediction
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      const rawText = textarea.value.trim();

      if (!rawText) {
        showAlert('Please paste or type a news article before analyzing.');
        textarea.focus();
        return;
      }

      const words = rawText.split(/\s+/).length;
      if (words < 5) {
        showAlert('The article is too short. Please provide at least 5 words for an accurate NLP assessment.');
        textarea.focus();
        return;
      }

      if (rawText.length > 40000) {
        showAlert('Article length exceeds the 40,000 character limit.');
        return;
      }

      if (validationAlert) validationAlert.style.display = 'none';
      if (resultsCard) resultsCard.style.display = 'none';
      if (loadingBox) loadingBox.style.display = 'block';

      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<span class="loading-spinner-sm"></span> Analyzing...';

      const pipelineSteps = [
        'Normalizing text and stripping wire datelines...',
        'Extracting TF-IDF unigram & bigram vectors...',
        'Running calibrated Machine Learning model...',
        'Synthesizing confidence & linguistic diagnostics...'
      ];

      let stepIdx = 0;
      const interval = setInterval(() => {
        stepIdx = (stepIdx + 1) % pipelineSteps.length;
        if (loadingStatusText) loadingStatusText.textContent = pipelineSteps[stepIdx];
      }, 300);

      let data = null;

      // First attempt: try Python Flask endpoint /predict
      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ text: rawText })
        });
        if (response.ok) {
          data = await response.json();
        }
      } catch (err) {
        // Backend not reachable (e.g. running on static GitHub Pages)
      }

      // Fallback: execute client-side ML engine directly in browser
      if (!data || !data.success) {
        try {
          data = await predictClientSide(rawText);
        } catch (clientErr) {
          clearInterval(interval);
          if (loadingBox) loadingBox.style.display = 'none';
          analyzeBtn.disabled = false;
          analyzeBtn.innerHTML = 'Analyze News';
          showAlert('Error analyzing article: ' + clientErr.message);
          return;
        }
      }

      clearInterval(interval);
      if (loadingBox) loadingBox.style.display = 'none';
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
        Analyze News
      `;

      // Save to local storage for GitHub Pages history
      saveToLocalHistory({
        text: rawText,
        label: data.label,
        confidence: data.confidence,
        real_prob: data.real_prob,
        fake_prob: data.fake_prob,
        timestamp: data.timestamp
      });

      // Render Results Card
      displayPredictionResult(data, rawText);
    });
  }

  function showAlert(msg) {
    if (validationAlert && alertMessage) {
      alertMessage.textContent = msg;
      validationAlert.style.display = 'flex';
    } else {
      alert(msg);
    }
  }

  function displayPredictionResult(res, originalText) {
    if (!resultsCard) return;

    const isReal = res.label === 'REAL';

    const resultBadge = document.getElementById('resultBadge');
    const statementHeading = document.getElementById('resultsStatementHeading');
    const circleBar = document.getElementById('circleProgressBar');
    const circleVal = document.getElementById('circlePercentVal');
    const probFillReal = document.getElementById('probFillReal');
    const probFillFake = document.getElementById('probFillFake');
    const probValReal = document.getElementById('probValReal');
    const probValFake = document.getElementById('probValFake');
    const explanationText = document.getElementById('explanationText');
    const signalWords = document.getElementById('signalWords');
    const signalUpper = document.getElementById('signalUpper');
    const signalExcl = document.getElementById('signalExcl');
    const signalAvgLen = document.getElementById('signalAvgLen');
    const analyzedPreview = document.getElementById('analyzedPreviewText');

    if (resultBadge) {
      resultBadge.className = `result-badge ${isReal ? 'result-badge-real' : 'result-badge-fake'}`;
      resultBadge.innerHTML = isReal
        ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> REAL NEWS`
        : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> FAKE NEWS`;
    }

    if (statementHeading) {
      statementHeading.textContent = isReal
        ? 'This article is predicted to be REAL.'
        : 'This article is predicted to be FAKE.';
    }

    const conf = res.confidence;
    const circumference = 440;
    const offset = circumference - (conf / 100) * circumference;

    if (circleBar) {
      circleBar.className = `circle-progress-bar ${isReal ? 'stroke-real' : 'stroke-fake'}`;
      circleBar.style.strokeDashoffset = '440';
      setTimeout(() => {
        circleBar.style.strokeDashoffset = `${offset}`;
      }, 50);
    }

    if (circleVal) {
      circleVal.textContent = `${conf}%`;
    }

    if (probFillReal) probFillReal.style.width = `${res.real_prob}%`;
    if (probFillFake) probFillFake.style.width = `${res.fake_prob}%`;
    if (probValReal) probValReal.textContent = `${res.real_prob}%`;
    if (probValFake) probValFake.textContent = `${res.fake_prob}%`;

    if (explanationText) explanationText.textContent = res.explanation;

    if (res.signals) {
      if (signalWords) signalWords.textContent = res.signals.word_count || 0;
      if (signalUpper) signalUpper.textContent = `${res.signals.uppercase_ratio || 0}%`;
      if (signalExcl) signalExcl.textContent = res.signals.exclamation_count || 0;
      if (signalAvgLen) signalAvgLen.textContent = `${res.signals.avg_word_length || 0} ch`;
    }

    if (analyzedPreview) {
      analyzedPreview.textContent = originalText;
    }

    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const copySummaryBtn = document.getElementById('copySummaryBtn');
    if (copySummaryBtn) {
      copySummaryBtn.onclick = () => {
        const summaryText = `[EchoCheck Analysis Result]\nPrediction: ${res.label}\nConfidence: ${res.confidence}%\nReal Probability: ${res.real_prob}%\nFake Probability: ${res.fake_prob}%\nTimestamp: ${res.timestamp}`;
        navigator.clipboard.writeText(summaryText).then(() => {
          copySummaryBtn.textContent = 'Copied to Clipboard!';
          setTimeout(() => { copySummaryBtn.textContent = 'Copy Summary'; }, 2000);
        });
      };
    }

    const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');
    if (analyzeAnotherBtn) {
      analyzeAnotherBtn.onclick = () => {
        textarea.value = '';
        updateCounters();
        resultsCard.style.display = 'none';
        textarea.focus();
        textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });
      };
    }
  }
}

/* ==========================================================================
   History Page Operations (Search, Filter, Modal, Clear History)
   ========================================================================== */
function initHistoryView() {
  const searchInput = document.getElementById('historySearch');
  const filterButtons = document.querySelectorAll('.filter-btn');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const confirmClearModal = document.getElementById('confirmClearModal');
  const viewArticleModal = document.getElementById('viewArticleModal');
  const cancelClearBtn = document.getElementById('cancelClearBtn');
  const executeClearBtn = document.getElementById('executeClearBtn');
  const closeArticleModalBtn = document.getElementById('closeArticleModalBtn');
  const fullArticleModalContent = document.getElementById('fullArticleModalContent');
  const historyTableBody = document.getElementById('historyTableBody');

  // Sync / populate from LocalStorage if on GitHub Pages (static mode)
  syncLocalStorageHistory();

  const historyRows = document.querySelectorAll('.history-row');

  // Search Filter
  if (searchInput) {
    searchInput.addEventListener('input', applyFilters);
  }

  // Tag Filter (All, Real, Fake)
  let activeTag = 'ALL';
  filterButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      filterButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTag = btn.getAttribute('data-filter') || 'ALL';
      applyFilters();
    });
  });

  function applyFilters() {
    const term = searchInput ? searchInput.value.toLowerCase().trim() : '';
    const currentRows = document.querySelectorAll('.history-row');

    currentRows.forEach(row => {
      const text = row.getAttribute('data-text') || '';
      const label = row.getAttribute('data-label') || '';

      const matchesSearch = !term || text.toLowerCase().includes(term) || label.toLowerCase().includes(term);
      const matchesTag = activeTag === 'ALL' || label.toUpperCase() === activeTag.toUpperCase();

      row.style.display = (matchesSearch && matchesTag) ? '' : 'none';
    });
  }

  // Clear History Modal
  if (clearHistoryBtn && confirmClearModal) {
    clearHistoryBtn.addEventListener('click', () => {
      confirmClearModal.classList.add('active');
    });

    if (cancelClearBtn) {
      cancelClearBtn.addEventListener('click', () => {
        confirmClearModal.classList.remove('active');
      });
    }

    if (executeClearBtn) {
      executeClearBtn.addEventListener('click', async () => {
        // Clear local storage
        localStorage.removeItem('echocheck_history');
        // Clear backend if running
        try {
          await fetch('/api/history', { method: 'DELETE' });
        } catch (e) {}
        window.location.reload();
      });
    }
  }

  // View Full Article Modal Trigger
  bindViewButtons();

  if (closeArticleModalBtn && viewArticleModal) {
    closeArticleModalBtn.addEventListener('click', () => {
      viewArticleModal.classList.remove('active');
    });
  }

  [confirmClearModal, viewArticleModal].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
      });
    }
  });

  initHistoryChart();
}

function bindViewButtons() {
  const viewArticleModal = document.getElementById('viewArticleModal');
  const fullArticleModalContent = document.getElementById('fullArticleModalContent');

  document.querySelectorAll('.btn-view-article').forEach(btn => {
    btn.onclick = (e) => {
      const fullText = e.currentTarget.getAttribute('data-full-text');
      if (viewArticleModal && fullArticleModalContent) {
        fullArticleModalContent.textContent = fullText;
        viewArticleModal.classList.add('active');
      }
    };
  });
}

function syncLocalStorageHistory() {
  const historyTableBody = document.getElementById('historyTableBody');
  if (!historyTableBody) return;

  const localHistory = JSON.parse(localStorage.getItem('echocheck_history') || '[]');
  const existingRows = historyTableBody.querySelectorAll('.history-row');

  // If local storage has items and table is either empty or we're on static page
  if (localHistory.length > 0 && existingRows.length === 0) {
    let rowsHTML = '';
    let realCount = 0;
    let fakeCount = 0;
    let totalConf = 0;

    localHistory.forEach((item, idx) => {
      const isReal = item.predicted_label === 'REAL';
      if (isReal) realCount++; else fakeCount++;
      totalConf += parseFloat(item.confidence_score || 0);

      rowsHTML += `
        <tr class="history-row" data-text="${(item.article_text || '').toLowerCase()}" data-label="${item.predicted_label}">
          <td style="color: var(--text-muted); font-size: 0.85rem; font-weight: 600;">#${idx + 1}</td>
          <td style="font-size: 0.85rem; color: var(--text-secondary); white-space: nowrap;">${item.created_at}</td>
          <td>
            <div class="table-snippet" title="${item.article_text}">
              ${item.article_snippet}
            </div>
          </td>
          <td>
            ${isReal
              ? `<span class="table-badge table-badge-real"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> REAL</span>`
              : `<span class="table-badge table-badge-fake"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> FAKE</span>`
            }
          </td>
          <td>
            <strong style="color: ${isReal ? 'var(--real-green-dark)' : 'var(--fake-red-dark)'};">
              ${item.confidence_score}%
            </strong>
          </td>
          <td style="text-align: right;">
            <button type="button" class="btn btn-secondary btn-sm btn-view-article" data-full-text="${item.article_text}" title="View Complete Text">
              <span>View</span>
            </button>
          </td>
        </tr>
      `;
    });

    historyTableBody.innerHTML = rowsHTML;

    // Update stats counters
    const total = localHistory.length;
    const realPct = total > 0 ? ((realCount / total) * 100).toFixed(1) : 0;
    const fakePct = total > 0 ? ((fakeCount / total) * 100).toFixed(1) : 0;
    const avgConf = total > 0 ? (totalConf / total).toFixed(1) : 0;

    const statTotalVal = document.getElementById('statTotalVal');
    const statRealPctVal = document.getElementById('statRealPctVal');
    const statFakePctVal = document.getElementById('statFakePctVal');
    const statAvgConfVal = document.getElementById('statAvgConfVal');

    if (statTotalVal) statTotalVal.textContent = total;
    if (statRealPctVal) statRealPctVal.textContent = `${realPct}%`;
    if (statFakePctVal) statFakePctVal.textContent = `${fakePct}%`;
    if (statAvgConfVal) statAvgConfVal.textContent = `${avgConf}%`;

    const chartCanvas = document.getElementById('historyDistributionChart');
    if (chartCanvas) {
      chartCanvas.setAttribute('data-real-count', realCount);
      chartCanvas.setAttribute('data-fake-count', fakeCount);
    }

    bindViewButtons();
  }
}

function initHistoryChart() {
  const canvas = document.getElementById('historyDistributionChart');
  if (!canvas || typeof Chart === 'undefined') return;

  const realCount = parseInt(canvas.getAttribute('data-real-count') || '0', 10);
  const fakeCount = parseInt(canvas.getAttribute('data-fake-count') || '0', 10);

  if (realCount === 0 && fakeCount === 0) return;

  new Chart(canvas, {
    type: 'doughnut',
    data: {
      labels: ['Real News', 'Fake News'],
      datasets: [{
        data: [realCount, fakeCount],
        backgroundColor: ['#10b981', '#ef4444'],
        hoverBackgroundColor: ['#059669', '#dc2626'],
        borderWidth: 2,
        borderColor: '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            boxWidth: 12,
            font: { family: 'Plus Jakarta Sans', size: 12 }
          }
        }
      }
    }
  });
}
