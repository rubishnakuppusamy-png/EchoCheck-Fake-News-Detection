/**
 * EchoCheck - AI-Based Fake News Detection System
 * Frontend Application Logic & Reactive Interactions
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

    // Hide validation alert as user types
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

  // 4. Form Submission & Real-Time Prediction
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      const rawText = textarea.value.trim();

      // Client-Side Validation
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

      // Hide results & alert, show loader
      if (validationAlert) validationAlert.style.display = 'none';
      if (resultsCard) resultsCard.style.display = 'none';
      if (loadingBox) loadingBox.style.display = 'block';

      analyzeBtn.disabled = true;
      analyzeBtn.innerHTML = '<span class="loading-spinner-sm"></span> Analyzing...';

      // Simulated pipeline progression steps for UX
      const pipelineSteps = [
        'Normalizing text and stripping wire prefixes...',
        'Extracting TF-IDF n-gram feature vectors...',
        'Running calibrated Machine Learning model...',
        'Synthesizing confidence & linguistic signals...'
      ];

      let stepIdx = 0;
      const interval = setInterval(() => {
        stepIdx = (stepIdx + 1) % pipelineSteps.length;
        if (loadingStatusText) loadingStatusText.textContent = pipelineSteps[stepIdx];
      }, 350);

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ text: rawText })
        });

        clearInterval(interval);
        const data = await response.json();

        if (loadingBox) loadingBox.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
          </svg>
          Analyze News
        `;

        if (!response.ok || !data.success) {
          showAlert(data.error || 'Server error occurred during prediction.');
          return;
        }

        // Render Results Card
        displayPredictionResult(data, rawText);

      } catch (err) {
        clearInterval(interval);
        if (loadingBox) loadingBox.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = 'Analyze News';
        showAlert('Network error: Unable to connect to the EchoCheck backend server. Please verify Flask is running.');
      }
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

    // 1. Badge & Statement
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

    // 2. Circular Meter Animation (circumference = 2 * PI * 70 = 439.82 ~ 440)
    const conf = res.confidence;
    const circumference = 440;
    const offset = circumference - (conf / 100) * circumference;

    if (circleBar) {
      circleBar.className = `circle-progress-bar ${isReal ? 'stroke-real' : 'stroke-fake'}`;
      // Trigger smooth transition
      circleBar.style.strokeDashoffset = '440';
      setTimeout(() => {
        circleBar.style.strokeDashoffset = `${offset}`;
      }, 50);
    }

    if (circleVal) {
      circleVal.textContent = `${conf}%`;
    }

    // 3. Dual Probability Bar
    if (probFillReal) probFillReal.style.width = `${res.real_prob}%`;
    if (probFillFake) probFillFake.style.width = `${res.fake_prob}%`;
    if (probValReal) probValReal.textContent = `${res.real_prob}%`;
    if (probValFake) probValFake.textContent = `${res.fake_prob}%`;

    // 4. Explanation & Signals
    if (explanationText) explanationText.textContent = res.explanation;

    if (res.signals) {
      if (signalWords) signalWords.textContent = res.signals.word_count || 0;
      if (signalUpper) signalUpper.textContent = `${res.signals.uppercase_ratio || 0}%`;
      if (signalExcl) signalExcl.textContent = res.signals.exclamation_count || 0;
      if (signalAvgLen) signalAvgLen.textContent = `${res.signals.avg_word_length || 0} ch`;
    }

    // 5. Analyzed Snippet
    if (analyzedPreview) {
      analyzedPreview.textContent = originalText;
    }

    // Show card and scroll
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Copy Summary Action
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

    // Analyze Another Button
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
  const historyRows = document.querySelectorAll('.history-row');
  const clearHistoryBtn = document.getElementById('clearHistoryBtn');
  const confirmClearModal = document.getElementById('confirmClearModal');
  const viewArticleModal = document.getElementById('viewArticleModal');
  const cancelClearBtn = document.getElementById('cancelClearBtn');
  const executeClearBtn = document.getElementById('executeClearBtn');
  const closeArticleModalBtn = document.getElementById('closeArticleModalBtn');
  const fullArticleModalContent = document.getElementById('fullArticleModalContent');

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

    historyRows.forEach(row => {
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
        try {
          const res = await fetch('/api/history', { method: 'DELETE' });
          const data = await res.json();
          if (data.success) {
            window.location.reload();
          } else {
            alert('Error clearing history: ' + (data.error || 'Unknown error'));
          }
        } catch (e) {
          alert('Network error while clearing history.');
        }
      });
    }
  }

  // View Full Article Modal Trigger
  document.querySelectorAll('.btn-view-article').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const fullText = e.currentTarget.getAttribute('data-full-text');
      if (viewArticleModal && fullArticleModalContent) {
        fullArticleModalContent.textContent = fullText;
        viewArticleModal.classList.add('active');
      }
    });
  });

  if (closeArticleModalBtn && viewArticleModal) {
    closeArticleModalBtn.addEventListener('click', () => {
      viewArticleModal.classList.remove('active');
    });
  }

  // Close modals on clicking overlay backdrop
  [confirmClearModal, viewArticleModal].forEach(modal => {
    if (modal) {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.remove('active');
      });
    }
  });

  // Chart.js Donut Chart Rendering (if canvas exists)
  initHistoryChart();
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
