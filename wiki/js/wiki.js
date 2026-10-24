/* HyperLithoMap Wiki — Navigation & interactivity */

(function () {
  'use strict';

  /* ── Active nav link ─────────────────────────────────────────────── */
  const current = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(a => {
    if (a.getAttribute('href') === current ||
        a.getAttribute('href') === './' + current) {
      a.classList.add('active');
    }
  });

  /* ── Mobile sidebar toggle ───────────────────────────────────────── */
  const toggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', e => {
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  /* ── Smooth-scroll heading anchors ──────────────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* ── Performance bars: animate on load ──────────────────────────── */
  const bars = document.querySelectorAll('.bar-fill[data-width]');
  if (bars.length) {
    setTimeout(() => {
      bars.forEach(b => { b.style.width = b.dataset.width; });
    }, 150);
  }

  /* ── Copy button for code blocks ────────────────────────────────── */
  document.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.style.cssText =
      'position:absolute;top:.6rem;right:.6rem;padding:.2rem .6rem;' +
      'font-size:.7rem;background:#243447;color:#e2e8f0;border:none;' +
      'border-radius:4px;cursor:pointer;font-family:inherit;';
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(pre.innerText.replace(/^Copy\n/, ''));
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    });
    pre.appendChild(btn);
  });

})();
