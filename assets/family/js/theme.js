import { renderThemeControl } from './theme-control.js';

/** Start each page with the system scheme; overrides last only for this page. */
export function initializeTheme({ root = document.documentElement, toggle } = {}) {
  const preference = window.matchMedia('(prefers-color-scheme: dark)');
  const compact = window.matchMedia('(max-width: 600px), (max-height: 420px)');
  let overridden = false;
  let dark = preference.matches;
  function apply() {
    root.dataset.theme = overridden ? (dark ? 'dark' : 'light') : 'system';
    if (toggle) renderThemeControl(toggle, dark);
  }
  function change() { overridden = !compact.matches; dark = overridden ? !dark : preference.matches; apply(); }
  function spaceChanged() { if (compact.matches) { overridden = false; dark = preference.matches; apply(); } }
  function systemChanged() { if (!overridden) { dark = preference.matches; apply(); } }
  function restorePage(event) { if (event.persisted) { overridden = false; dark = preference.matches; apply(); } }
  apply();
  toggle?.addEventListener('click', change);
  preference.addEventListener('change', systemChanged);
  compact.addEventListener('change', spaceChanged);
  window.addEventListener('pageshow', restorePage);
  return { destroy() { toggle?.removeEventListener('click', change); preference.removeEventListener('change', systemChanged); compact.removeEventListener('change', spaceChanged); window.removeEventListener('pageshow', restorePage); } };
}
