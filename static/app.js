'use strict';

// ── i18n ──────────────────────────────────────────────────────────────────
const LANG = {
  en: {
    title:            'FeelText',
    subtitle:         'Multilingual Sentiment Analysis',
    tab_analyze:      'Analyze',
    tab_info:         'API Info',
    analyze_title:    'Analyze Text',
    analyze_desc:     'Enter any text to detect its sentiment',
    analyze_ph:       'Type or paste your text here…',
    analyze_btn:      'Analyze',
    loading:          'Loading…',
    health_ok:        'Model Ready',
    health_nok:       'Model Not Ready',
    health_offline:   'API Offline',
    s_positive:       'Positive',
    s_negative:       'Negative',
    s_neutral:        'Neutral',
    confidence:       'Conf.',
    lang_label:       'Language',
    lang_english:     'English',
    lang_persian:     'Persian',
    lang_other:       'Other',
    text_lbl:         'Text',
    err_empty:        'Please enter some text first.',
    err_api:          'Could not reach the API.',
    health_title:     '📊 API Health',
    status_lbl:       'Status',
    status_ok:        'Healthy',
    status_err:       'Error',
    mode_lbl:         'Mode',
    mode_custom:      'Custom Model (TF-IDF + Naive Bayes)',
    mode_third:       'Third-party (HuggingFace)',
    model_lbl:        'Model',
    langs_title:      '🌐 Supported Languages',
    footer_text:      'FeelText — Multilingual Sentiment Analysis API',
    footer_docs:      'Swagger Docs',
    hint:             'Tip: Ctrl+Enter to analyze quickly',
    tooltip_model:    'Model',
    tooltip_desc_custom: 'Custom TF-IDF vectorizer + Complement Naive Bayes, trained from scratch on the IMDB dataset.',
    tooltip_desc_third:  'Pre-trained multilingual transformers via HuggingFace (BERT-based).',
    tooltip_desc_unknown: 'Sentiment analysis model.',
  },
  fa: {
    title:            'FeelText',
    subtitle:         'تحلیل احساسات چندزبانه',
    tab_analyze:      'تحلیل',
    tab_info:         'اطلاعات API',
    analyze_title:    'تحلیل متن',
    analyze_desc:     'هر متنی را وارد کنید تا احساسات آن تشخیص داده شود',
    analyze_ph:       'متن خود را اینجا بنویسید…',
    analyze_btn:      'تحلیل کن',
    loading:          'در حال بارگذاری…',
    health_ok:        'مدل آماده است',
    health_nok:       'مدل آماده نیست',
    health_offline:   'API آفلاین است',
    s_positive:       'مثبت',
    s_negative:       'منفی',
    s_neutral:        'خنثی',
    confidence:       'اطمینان',
    lang_label:       'زبان',
    lang_english:     'انگلیسی',
    lang_persian:     'فارسی',
    lang_other:       'سایر',
    text_lbl:         'متن',
    err_empty:        'لطفاً ابتدا متنی وارد کنید.',
    err_api:          'اتصال به API برقرار نشد.',
    health_title:     '📊 وضعیت API',
    status_lbl:       'وضعیت',
    status_ok:        'سالم',
    status_err:       'خطا',
    mode_lbl:         'حالت',
    mode_custom:      'مدل سفارشی (TF-IDF + Naive Bayes)',
    mode_third:       'مدل پیش‌آموزش‌دیده (HuggingFace)',
    model_lbl:        'مدل',
    langs_title:      '🌐 زبان‌های پشتیبانی‌شده',
    footer_text:      'فیل‌تکست — API تحلیل احساسات چندزبانه',
    footer_docs:      'مستندات Swagger',
    hint:             'نکته: Ctrl+Enter برای تحلیل سریع',
    tooltip_model:    'مدل',
    tooltip_desc_custom: 'وکتوریزر TF-IDF سفارشی + Naive Bayes مکمل، آموزش‌دیده از صفر روی داده‌های IMDB.',
    tooltip_desc_third:  'مدل‌های زبانی چندزبانه پیش‌آموزش‌دیده از HuggingFace (مبتنی بر BERT).',
    tooltip_desc_unknown: 'مدل تحلیل احساسات.',
  },
};

// ── State ──────────────────────────────────────────────────────────────────
let lang   = localStorage.getItem('feeltext-lang') || 'en';
let health = null;

// ── Sentiment config ───────────────────────────────────────────────────────
const SC = {
  positive: { emoji: '😊', color: '#22c55e', key: 's_positive' },
  negative: { emoji: '😔', color: '#ef4444', key: 's_negative' },
  neutral:  { emoji: '😐', color: '#f59e0b', key: 's_neutral'  },
};

// ── Helpers ────────────────────────────────────────────────────────────────
const tr = key => LANG[lang][key] || key;

function esc(s) {
  if (!s) return '';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function clip(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : (s || ''); }

function langLabel(code) {
  if (!code) return '—';
  const m = { persian: 'lang_persian', farsi: 'lang_persian', english: 'lang_english' };
  const key = m[code.toLowerCase()];
  return key ? tr(key) : `${tr('lang_other')} (${code})`;
}

function showToast(msg, type = 'info') {
  document.querySelector('.toast')?.remove();
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('show')));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 350); }, 3200);
}

// ── Language toggle ────────────────────────────────────────────────────────
function toggleLang() {
  lang = lang === 'en' ? 'fa' : 'en';
  localStorage.setItem('feeltext-lang', lang);
  applyLang();
}

function applyLang() {
  const isRTL = lang === 'fa';
  document.documentElement.lang = lang;
  document.documentElement.dir  = isRTL ? 'rtl' : 'ltr';
  document.getElementById('lang-btn').textContent = isRTL ? 'EN' : 'فا';

  // Update all static i18n nodes, but skip the health-text which is managed by fetchHealth/updateHealthBadgeText
  document.querySelectorAll('[data-i18n]').forEach(el => {
    if (el.classList.contains('health-text')) return;
    const v = LANG[lang][el.dataset.i18n];
    if (typeof v === 'string') el.textContent = v;
  });

  document.querySelectorAll('[data-i18n-ph]').forEach(el => {
    const v = LANG[lang][el.dataset.i18nPh];
    if (v) el.placeholder = typeof v === 'string' ? v : v(1);
  });

  // Update rendered sentiment labels
  document.querySelectorAll('.result-card[data-s]').forEach(card => {
    const s = card.dataset.s;
    const cfg = SC[s];
    if (cfg) card.querySelector('.rc-sentiment').textContent = tr(cfg.key);
    card.querySelectorAll('.score-lbl[data-sk]').forEach(el => {
      el.textContent = tr(SC[el.dataset.sk]?.key || el.dataset.sk);
    });
  });

  // Re-apply correct health badge text and tooltip in the current language
  updateHealthBadgeText();
  updateTooltip();

  if (document.getElementById('tab-info').classList.contains('active')) {
    renderInfo();
  }
}

// ── Tabs ───────────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
  if (name === 'info') renderInfo();
}

// ── Text direction auto-detection ──────────────────────────────────────────
function detectDir(el) {
  const text = el.value;
  if (!text) return;
  const fa = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const en = (text.match(/[a-zA-Z]/g) || []).length;
  const total = fa + en;
  if (total) el.dir = fa / total > 0.3 ? 'rtl' : 'ltr';
}

function onInputChange(el) {
  detectDir(el);
  const n = el.value.length;
  document.getElementById('char-count').textContent = `${n} / 5000`;
}

// ── API ────────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Health badge text ──────────────────────────────────────────────────────
function updateHealthBadgeText() {
  const badge = document.getElementById('health-badge');
  const txt   = badge.querySelector('.health-text');
  if (!txt) return;

  if (health === null) {
    // fetchHealth hasn't completed yet — keep showing translated "Loading…"
    txt.textContent = tr('loading');
  } else if (health === undefined) {
    badge.className = 'health-badge err';
    txt.textContent = tr('health_offline');
  } else {
    const loaded = health.model_loaded;
    badge.className = `health-badge ${loaded ? 'ok' : 'err'}`;
    txt.textContent = loaded ? tr('health_ok') : tr('health_nok');
  }
}

// ── Tooltip ────────────────────────────────────────────────────────────────
function updateTooltip() {
  const nameEl = document.getElementById('tooltip-name');
  const descEl = document.getElementById('tooltip-desc');
  if (!nameEl || !descEl) return;

  if (!health || health === undefined) {
    nameEl.textContent = '—';
    descEl.textContent = '—';
    return;
  }

  nameEl.textContent = health.model_name || '—';

  const descKey = health.mode === 'custom'
    ? 'tooltip_desc_custom'
    : health.mode === 'third-party'
      ? 'tooltip_desc_third'
      : 'tooltip_desc_unknown';

  descEl.textContent = tr(descKey);
}

// ── Health ─────────────────────────────────────────────────────────────────
async function fetchHealth() {
  try {
    health = await api('/health');
  } catch {
    health = undefined;
  }
  updateHealthBadgeText();
  updateTooltip();
}

// ── Analyze single ─────────────────────────────────────────────────────────
async function doAnalyze() {
  const input = document.getElementById('analyze-input');
  const btn   = document.getElementById('analyze-btn');
  const area  = document.getElementById('single-result');
  const text  = input.value.trim();

  if (!text) {
    input.classList.add('shake');
    setTimeout(() => input.classList.remove('shake'), 500);
    showToast(tr('err_empty'), 'error');
    return;
  }

  btn.disabled = true;
  btn.classList.add('busy');

  try {
    const result = await api('/analyze', { method: 'POST', body: JSON.stringify({ text }) });
    area.innerHTML = buildCard(result);
    animateBars(area);
    animateRing(area.querySelector('.conf-ring'), result.confidence, SC[result.sentiment]?.color || '#6366f1');
    area.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch {
    showToast(tr('err_api'), 'error');
  } finally {
    btn.disabled = false;
    btn.classList.remove('busy');
  }
}

// ── Card builder ───────────────────────────────────────────────────────────
function buildCard(result) {
  const { sentiment, confidence, scores, language_detected, text } = result;
  const cfg = SC[sentiment] || SC.neutral;
  const pct = Math.round(confidence * 100);

  const scoreBars = ['positive', 'negative', 'neutral'].map(type => {
    const val   = scores[type] || 0;
    const p     = Math.round(val * 100);
    const color = SC[type]?.color || '#6366f1';
    return `
      <div class="score-row">
        <span class="score-lbl" data-sk="${type}" style="color:${color}">${tr(SC[type]?.key || type)}</span>
        <div class="score-track">
          <div class="score-fill" style="background:${color}" data-target="${p}"></div>
        </div>
        <span class="score-pct">${p}%</span>
      </div>`;
  }).join('');

  return `
    <div class="result-card ${sentiment}" data-s="${sentiment}" data-conf="${confidence}">
      <div class="rc-bar"></div>
      <div class="rc-body">
        <div class="rc-head">
          <div class="rc-emoji">${cfg.emoji}</div>
          <div class="rc-labels">
            <div class="rc-sentiment">${tr(cfg.key)}</div>
            <div class="rc-lang">🌐 ${langLabel(language_detected)}</div>
          </div>
          <div class="conf-ring" data-target="${pct}" style="--ring-c:${cfg.color}">
            <div class="conf-text">
              <div class="conf-pct">${pct}%</div>
              <div class="conf-lbl">${tr('confidence')}</div>
            </div>
          </div>
        </div>
        <div class="score-bars">${scoreBars}</div>
        <div class="rc-meta">
          <span class="rc-badge">📝 ${tr('text_lbl')}</span>
          <span class="rc-text-preview" title="${esc(text)}">${esc(clip(text, 90))}</span>
        </div>
      </div>
    </div>`;
}

// ── Animations ─────────────────────────────────────────────────────────────
function animateBars(container) {
  requestAnimationFrame(() => {
    setTimeout(() => {
      container.querySelectorAll('.score-fill').forEach(bar => {
        bar.style.width = bar.dataset.target + '%';
      });
    }, 90);
  });
}

function animateRing(ring, confidence, color) {
  if (!ring) return;
  const target = Math.round(confidence * 100);
  let cur = 0;
  const step = () => {
    cur = Math.min(cur + 2, target);
    ring.style.background = `conic-gradient(${color} 0% ${cur}%, rgba(255,255,255,.07) 0%)`;
    if (cur < target) requestAnimationFrame(step);
  };
  setTimeout(() => requestAnimationFrame(step), 100);
}

// ── Info tab ───────────────────────────────────────────────────────────────
async function renderInfo() {
  const el = document.getElementById('info-content');
  el.innerHTML = `<div class="loading-msg">${tr('loading')}</div>`;

  let langs = [];
  try { langs = (await api('/languages')).languages || []; } catch { /* ignore */ }

  const h         = health && health !== undefined ? health : null;
  const isOk      = h?.model_loaded ?? false;
  const mode      = h?.mode || '—';
  const model     = h?.model_name || '—';
  const modeLabel = mode === 'custom'
    ? tr('mode_custom')
    : mode === 'third-party'
      ? tr('mode_third')
      : mode === '—' ? '—' : mode;

  el.innerHTML = `
    <div class="info-grid">
      <div class="info-card">
        <h3>${tr('health_title')}</h3>
        <div class="info-row">
          <span class="info-key">${tr('status_lbl')}</span>
          <span class="info-val">
            <span class="status-dot ${isOk ? 'ok' : 'err'}"></span>${isOk ? tr('status_ok') : tr('status_err')}
          </span>
        </div>
        <div class="info-row">
          <span class="info-key">${tr('mode_lbl')}</span>
          <span class="info-val">${esc(modeLabel)}</span>
        </div>
        <div class="info-row">
          <span class="info-key">${tr('model_lbl')}</span>
          <span class="info-val" style="font-size:.75rem;color:var(--muted)">${esc(model)}</span>
        </div>
      </div>
      <div class="info-card">
        <h3>${tr('langs_title')}</h3>
        <div class="lang-chips">
          ${langs.map(l => `<span class="lang-chip">${esc(l)}</span>`).join('') ||
            `<span style="color:var(--faint);font-size:.85rem">—</span>`}
        </div>
      </div>
    </div>`;
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('analyze-input');

  input.addEventListener('input', () => onInputChange(input));
  input.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') doAnalyze();
  });

  applyLang();
  fetchHealth();
});
