import { button } from './workspace.js';

export function initializeBackTop({ focusTarget } = {}) {
  const control = button('', () => {
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    focusTarget?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: reduced ? 'instant' : 'smooth' });
  }, 'isi-button isi-button--primary isi-button--icon workspace-back-top');
  control.setAttribute('aria-label', 'Back to top');
  control.dataset.state = 'hidden'; control.inert = true;
  control.setAttribute('aria-hidden', 'true');
  control.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="m5 12 7-7 7 7M12 5v14"/></svg>';
  document.body.append(control);
  const update = () => {
    const hidden = window.scrollY <= document.documentElement.clientHeight;
    const state = hidden ? 'hidden' : 'visible';
    if (control.dataset.state === state) return;
    control.dataset.state = state;
    control.inert = hidden;
    control.setAttribute('aria-hidden', String(hidden));
  };
  window.addEventListener('scroll', update, { passive: true });
  window.addEventListener('resize', update); update();
  return { destroy() { window.removeEventListener('scroll', update); window.removeEventListener('resize', update); control.remove(); } };
}
