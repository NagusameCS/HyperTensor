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
    // canonical reading surface. Compiled PDFs are linked from the
    // research-papers tab on the home page.)
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', buildPanel);
  } else {
    buildPanel();
  }

  // ─── In-page link previews + broken multi-target anchor repair ──────
  // Pandoc renders \Cref{a,b,c} as a single anchor href="#a,b,c" which
  // resolves to nothing. We split such anchors into individual links at
  // load time so the rendered papers always navigate correctly even if
  // the source LaTeX uses multi-target \Cref. We then attach a small
  // hover preview that shows the heading + first paragraph of the
  // target section.
  function labelFromId(id) {
    if (!id) return '';
    var s = id.replace(/^sec:|^app:|^tab:|^fig:/, '');
    s = s.replace(/[-_]+/g, ' ');
    return s.replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function repairMultiAnchors(root) {
    var anchors = root.querySelectorAll('a[href^="#"]');
    [].forEach.call(anchors, function (a) {
      var href = a.getAttribute('href') || '';
      if (href.indexOf(',') === -1) return;
      var ids = href.slice(1).split(',').map(function (s) { return s.trim(); }).filter(Boolean);
      if (ids.length < 2) return;
      var frag = document.createDocumentFragment();
      ids.forEach(function (id, i) {
        var aa = document.createElement('a');
        aa.href = '#' + id;
        aa.setAttribute('data-reference', id);
        aa.setAttribute('data-reference-type', 'ref');
        aa.textContent = '\u00a7' + labelFromId(id);
        frag.appendChild(aa);
        if (i < ids.length - 1) {
          frag.appendChild(document.createTextNode(i === ids.length - 2 ? ', and ' : ', '));
        }
      });
      a.parentNode.replaceChild(frag, a);
    });
  }

  function getPreviewFor(id) {
    var t = document.getElementById(id);
    if (!t) return null;
    // Use heading text if target is a section
    var heading = t.querySelector('h1,h2,h3,h4,h5,h6');
    var title = heading ? heading.textContent.trim() : (t.getAttribute('aria-label') || labelFromId(id));
    // First paragraph after the heading (or inside the target itself)
    var p = t.querySelector('p');
    var snippet = p ? p.textContent.trim().slice(0, 280) : '';
    if (snippet.length === 280) snippet += '\u2026';
    var label = 'In this paper';
    if (/^tab:/.test(id)) label = 'Table';
    else if (/^fig:/.test(id)) label = 'Figure';
    else if (/^app:/.test(id)) label = 'Appendix';
    else if (/^sec:/.test(id)) label = 'Section';
    return { label: label, title: title, snippet: snippet };
  }

  function buildPreviewEl() {
    var el = document.getElementById('ht-link-preview');
    if (el) return el;
    // Inject styles once. Mirrors the rule set in paper-style.css so the
    // tooltip works on pages that don't load paper-style.css (e.g. home).
    if (!document.getElementById('ht-link-preview-style')) {
      var st = document.createElement('style');
      st.id = 'ht-link-preview-style';
      st.textContent = [
        '#ht-link-preview{position:fixed;z-index:9999;max-width:380px;min-width:220px;',
        'padding:.7rem .85rem;background:#fff;border:1px solid rgba(0,0,0,.12);',
        'border-radius:8px;box-shadow:0 8px 28px rgba(0,0,0,.12);font-size:.84rem;',
        'line-height:1.45;color:#1a1a1a;pointer-events:none;opacity:0;',
        'transform:translateY(-2px);transition:opacity .12s ease,transform .12s ease;',
        'font-family:inherit;}',
        '#ht-link-preview[data-open="1"]{opacity:1;transform:translateY(0);}',
        '#ht-link-preview .ht-lp-label{display:block;font-size:.68rem;font-weight:700;',
        'letter-spacing:.08em;text-transform:uppercase;color:#666;margin-bottom:.3rem;}',
        '#ht-link-preview .ht-lp-title{display:block;font-weight:600;margin-bottom:.3rem;}',
        '#ht-link-preview .ht-lp-snippet{color:#555;font-size:.82rem;}',
        '@media (prefers-color-scheme: dark){',
        ' :root:not([data-theme="light"]):not([data-theme="sepia"]) #ht-link-preview',
        '   {background:#1c1c1c;color:#e8e8e8;border-color:rgba(255,255,255,.12);}',
        ' :root:not([data-theme="light"]):not([data-theme="sepia"]) #ht-link-preview .ht-lp-label{color:#9a9a9a;}',
        ' :root:not([data-theme="light"]):not([data-theme="sepia"]) #ht-link-preview .ht-lp-snippet{color:#bdbdbd;}',
        '}',
        '[data-theme="dark"] #ht-link-preview{background:#1c1c1c;color:#e8e8e8;border-color:rgba(255,255,255,.12);}',
        '[data-theme="dark"] #ht-link-preview .ht-lp-label{color:#9a9a9a;}',
        '[data-theme="dark"] #ht-link-preview .ht-lp-snippet{color:#bdbdbd;}'
      ].join('');
      document.head.appendChild(st);
    }
    el = document.createElement('div');
    el.id = 'ht-link-preview';
    el.setAttribute('role', 'tooltip');
    el.innerHTML = '<span class="ht-lp-label"></span><span class="ht-lp-title"></span><span class="ht-lp-snippet"></span>';
    document.body.appendChild(el);
    return el;
  }

  function placePreview(el, mouseX, mouseY) {
    var pad = 14;
    var rect = el.getBoundingClientRect();
    var vw = window.innerWidth, vh = window.innerHeight;
    var x = mouseX + 16;
    var y = mouseY + 16;
    if (x + rect.width + pad > vw) x = mouseX - rect.width - 16;
    if (y + rect.height + pad > vh) y = mouseY - rect.height - 16;
    if (x < pad) x = pad;
    if (y < pad) y = pad;
    el.style.left = x + 'px';
    el.style.top  = y + 'px';
  }

  function attachLinkPreviews(root) {
    var el = buildPreviewEl();
    var hideTimer = null;
    var showTimer = null;

    function show(a, evt) {
      var href = a.getAttribute('href') || '';
      var data = null;
      if (href.charAt(0) === '#' && href.length > 1) {
        data = getPreviewFor(href.slice(1));
      } else if (/^https?:\/\//.test(href)) {
        var url;
        try { url = new URL(href); } catch (e) { return; }
        data = { label: 'External link', title: url.hostname.replace(/^www\./, ''), snippet: url.pathname + (url.search || '') };
      }
      if (!data) return;
      el.querySelector('.ht-lp-label').textContent   = data.label;
      el.querySelector('.ht-lp-title').textContent   = data.title;
      el.querySelector('.ht-lp-snippet').textContent = data.snippet;
      el.setAttribute('data-open', '1');
      placePreview(el, evt.clientX || 0, evt.clientY || 0);
    }
    function hide() { el.setAttribute('data-open', '0'); }

    root.addEventListener('mouseover', function (e) {
      var a = e.target && e.target.closest && e.target.closest('a[href]');
      if (!a) return;
      // skip nav, reader-toggle, page-internal nav we don't want to preview
      if (a.closest('.nav-links') || a.classList.contains('reader-toggle')) return;
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = setTimeout(function () { show(a, e); }, 280);
    });
    root.addEventListener('mousemove', function (e) {
      if (el.getAttribute('data-open') === '1') placePreview(el, e.clientX, e.clientY);
    });
    root.addEventListener('mouseout', function (e) {
      var a = e.target && e.target.closest && e.target.closest('a[href]');
      if (!a) return;
      clearTimeout(showTimer);
      hideTimer = setTimeout(hide, 80);
    });
    // hide on scroll/resize/keypress to stay unobtrusive
    window.addEventListener('scroll', hide, { passive: true });
    window.addEventListener('resize', hide);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });
  }

  function initLinkUx() {
    repairMultiAnchors(document);
    // Skip the heavy hover handler on small screens (touch) to avoid
    // accidental triggers from tap-and-hold.
    if (window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches) {
      attachLinkPreviews(document);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLinkUx);
  } else {
    initLinkUx();
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
