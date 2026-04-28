/* reader.js — injects the floating reader-controls panel and persists
   the user's choices to localStorage. Loaded on every paper page. */
(function () {
  'use strict';

  var KEY = 'hypertensor.reader.v1';
  var ATTRS = ['theme', 'fontsize', 'lineheight', 'width', 'font', 'graphs'];

  var DEFAULTS = {
    theme: null,        // null = follow prefers-color-scheme
    fontsize: 'm',
    lineheight: 'normal',
    width: 'normal',
    font: 'sans',
    graphs: 'static' // static by default; user can opt into interactive graphs
  };

  function load() {
    try {
      var raw = localStorage.getItem(KEY);
      if (!raw) return Object.assign({}, DEFAULTS);
      return Object.assign({}, DEFAULTS, JSON.parse(raw));
    } catch (e) { return Object.assign({}, DEFAULTS); }
  }

  function save(s) {
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {}
  }

  function effectiveTheme(s) {
    if (s.theme === 'light' || s.theme === 'dark' || s.theme === 'sepia') return s.theme;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
    return 'light';
  }

  function apply(s) {
    var html = document.documentElement;
    var theme = effectiveTheme(s);
    if (theme === 'light') html.removeAttribute('data-theme');
    else html.setAttribute('data-theme', theme);
    html.setAttribute('data-fontsize', s.fontsize);
    html.setAttribute('data-lineheight', s.lineheight);
    html.setAttribute('data-width', s.width);
    html.setAttribute('data-font', s.font);
    html.setAttribute('data-graphs', s.graphs || DEFAULTS.graphs);
  }

  function emitChange(s) {
    try {
      document.dispatchEvent(new CustomEvent('hypertensor:reader-settings-changed', {
        detail: { state: Object.assign({}, s) }
      }));
    } catch (e) {}
  }

  // Apply BEFORE DOM ready to avoid flash on dark mode.
  var state = load();
  apply(state);

  // Public API so page-level scripts can query reader preferences.
  window.HyperTensorReader = {
    getState: function () { return Object.assign({}, state); },
    interactiveGraphsEnabled: function () { return (state.graphs || DEFAULTS.graphs) === 'interactive'; }
  };

  // Favicon: site-wide GitHub avatar. Injected once if absent so we don't
  // have to thread a <link rel="icon"> through every HTML file.
  function ensureFavicon() {
    if (!document || !document.head) return;
    if (document.querySelector('link[rel="icon"]')) return;
    var l = document.createElement('link');
    l.rel = 'icon';
    l.type = 'image/png';
    l.href = 'https://github.com/NagusameCS.png';
    document.head.appendChild(l);
  }
  ensureFavicon();

  function buildPanel() {
    var toggle = document.createElement('button');
    toggle.className = 'reader-toggle';
    toggle.setAttribute('aria-label', 'Reader settings');
    toggle.title = 'Reader settings';
    toggle.innerHTML = '\u00b6'; // pilcrow

    var panel = document.createElement('div');
    panel.className = 'reader-panel';
    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-label', 'Reader settings');

    function row(label, key, opts) {
      var h = document.createElement('h4');
      h.textContent = label;
      panel.appendChild(h);
      var r = document.createElement('div');
      r.className = 'reader-row';
      opts.forEach(function (opt) {
        var b = document.createElement('button');
        b.className = 'reader-btn';
        b.type = 'button';
        b.textContent = opt.label;
        b.setAttribute('data-key', key);
        b.setAttribute('data-value', opt.value);
        b.setAttribute('aria-pressed', String((state[key] || (key === 'theme' ? null : DEFAULTS[key])) === opt.value));
        b.addEventListener('click', function () {
          state[key] = opt.value;
          apply(state);
          save(state);
          emitChange(state);
          // refresh pressed states for this row
          [].forEach.call(r.querySelectorAll('.reader-btn'), function (sib) {
            sib.setAttribute('aria-pressed', String(sib.getAttribute('data-value') === opt.value));
          });
        });
        r.appendChild(b);
      });
      panel.appendChild(r);
    }

    row('Theme',   'theme',   [
      { label: 'Auto',  value: 'auto' },
      { label: 'Light', value: 'light' },
      { label: 'Sepia', value: 'sepia' },
      { label: 'Dark',  value: 'dark' }
    ]);
    row('Font size',   'fontsize', [
      { label: 'XS', value: 'xs' },
      { label: 'S',  value: 's' },
      { label: 'M',  value: 'm' },
      { label: 'L',  value: 'l' },
      { label: 'XL', value: 'xl' },
      { label: 'XXL',value: 'xxl' }
    ]);
    row('Line height', 'lineheight', [
      { label: 'Tight',  value: 'tight' },
      { label: 'Normal', value: 'normal' },
      { label: 'Loose',  value: 'loose' }
    ]);
    row('Column width', 'width', [
      { label: 'Narrow', value: 'narrow' },
      { label: 'Normal', value: 'normal' },
      { label: 'Wide',   value: 'wide' },
      { label: 'Full',   value: 'full' }
    ]);
    row('Font family', 'font', [
      { label: 'Sans',  value: 'sans' },
      { label: 'Serif', value: 'serif' },
      { label: 'Mono',  value: 'mono' }
    ]);
    row('Graphs', 'graphs', [
      { label: 'Static',      value: 'static' },
      { label: 'Interactive', value: 'interactive' }
    ]);

    // Reset
    var hReset = document.createElement('h4');
    hReset.textContent = 'Reset';
    panel.appendChild(hReset);
    var rReset = document.createElement('div');
    rReset.className = 'reader-row';
    var bReset = document.createElement('button');
    bReset.className = 'reader-btn';
    bReset.type = 'button';
    bReset.textContent = 'Restore defaults';
    bReset.addEventListener('click', function () {
      state = Object.assign({}, DEFAULTS, { theme: 'auto' });
      apply(state);
      save(state);
      emitChange(state);
      // refresh all pressed states
      [].forEach.call(panel.querySelectorAll('.reader-btn'), function (b) {
        var k = b.getAttribute('data-key');
        var v = b.getAttribute('data-value');
        if (k) {
          var current = state[k];
          if (k === 'theme' && (state.theme == null || state.theme === 'auto')) current = 'auto';
          b.setAttribute('aria-pressed', String(current === v));
        } else {
          b.removeAttribute('aria-pressed');
        }
      });
    });
    rReset.appendChild(bReset);
    panel.appendChild(rReset);

    // Map "auto" back to null when persisting (so prefers-color-scheme drives)
    panel.addEventListener('click', function (e) {
      if (e.target && e.target.getAttribute('data-key') === 'theme' &&
          e.target.getAttribute('data-value') === 'auto') {
        state.theme = null;
        apply(state);
        save(state);
        emitChange(state);
      }
    });

    toggle.addEventListener('click', function () {
      var open = panel.getAttribute('data-open') === '1';
      panel.setAttribute('data-open', open ? '0' : '1');
    });

    document.addEventListener('click', function (e) {
      if (panel.getAttribute('data-open') !== '1') return;
      if (panel.contains(e.target) || toggle.contains(e.target)) return;
      panel.setAttribute('data-open', '0');
    });

    document.body.appendChild(toggle);
    document.body.appendChild(panel);

    // (Print/Save-as-PDF button removed; the on-screen renders are the
    // canonical reading surface. arXiv-style PDFs are linked from the
    // research-papers tab on the home page.)
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildPanel);
  } else {
    buildPanel();
  }

  // React to system theme changes when in auto mode
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var listener = function () {
      if (state.theme == null || state.theme === 'auto') apply(state);
    };
    if (mq.addEventListener) mq.addEventListener('change', listener);
    else if (mq.addListener) mq.addListener(listener);
  }
})();
