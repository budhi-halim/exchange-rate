import { button } from './workspace.js';

export function initializeBackTop({ focusTarget } = {}) {
  const events = new AbortController();
  let pointer = null;
  let suppressClick = false;
  let settlingFrame = 0;
  function cancelSettle() {
    cancelAnimationFrame(settlingFrame);
    settlingFrame = 0;
  }
  function settleTouchReturn() {
    // The compositor can deliver a final fling frame after an instant scroll.
    let framesLeft = 2;
    const settle = () => {
      settlingFrame = 0;
      if (window.scrollY > 0) window.scrollTo({ top: 0, behavior: 'instant' });
      if (--framesLeft > 0) settlingFrame = requestAnimationFrame(settle);
    };
    settlingFrame = requestAnimationFrame(settle);
  }
  function activate({ touch = false, keyboard = false } = {}) {
    cancelSettle();
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (keyboard) focusTarget?.focus({ preventScroll: true });
    // A touch return skips intermediate off-screen content and cancels momentum.
    window.scrollTo({ top: 0, behavior: touch || reduced ? 'instant' : 'smooth' });
    if (touch) settleTouchReturn();
  }
  const control = button('', event => {
    const handled = suppressClick;
    suppressClick = false;
    if (handled && event.detail > 0) { event.preventDefault(); return; }
    const touch = ['touch', 'pen'].includes(event.pointerType) || (event.detail > 0 && matchMedia('(hover: none) and (pointer: coarse)').matches);
    activate({ touch, keyboard: event.detail === 0 && !touch });
  }, 'isi-button isi-button--primary isi-button--icon workspace-back-top');
  control.setAttribute('aria-label', 'Back to top');
  control.dataset.state = 'hidden'; control.inert = true;
  control.setAttribute('aria-hidden', 'true');
  control.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="m5 12 7-7 7 7M12 5v14"/></svg>';
  document.body.append(control);
  const signal = events.signal;
  for (const name of ['pointerdown', 'touchstart', 'wheel', 'keydown']) {
    window.addEventListener(name, cancelSettle, { capture: true, passive: true, signal });
  }
  function clearPointer() {
    const previous = pointer;
    pointer = null;
    if (previous && control.hasPointerCapture(previous.id)) control.releasePointerCapture(previous.id);
  }
  control.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button !== 0) return;
    suppressClick = false;
    if (!['touch', 'pen'].includes(event.pointerType)) return;
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    control.setPointerCapture(event.pointerId);
    event.preventDefault();
    // Handle the gesture before a fling can consume its compatibility click.
    window.scrollTo({ top: window.scrollY, left: window.scrollX, behavior: 'instant' });
  }, { signal });
  control.addEventListener('pointermove', event => {
    if (pointer?.id === event.pointerId && Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 12) pointer.moved = true;
  }, { passive: true, signal });
  control.addEventListener('pointerup', event => {
    if (pointer?.id !== event.pointerId) return;
    const moved = pointer.moved || Math.hypot(event.clientX - pointer.x, event.clientY - pointer.y) > 12;
    clearPointer();
    suppressClick = true;
    event.preventDefault();
    if (!moved) activate({ touch: true });
  }, { signal });
  for (const name of ['pointercancel', 'lostpointercapture']) control.addEventListener(name, clearPointer, { signal });
  window.addEventListener('pagehide', () => { cancelSettle(); clearPointer(); suppressClick = false; }, { signal });
  const update = () => {
    const hidden = window.scrollY <= window.innerHeight;
    const state = hidden ? 'hidden' : 'visible';
    if (control.dataset.state === state) return;
    control.dataset.state = state;
    control.inert = hidden;
    control.setAttribute('aria-hidden', String(hidden));
  };
  window.addEventListener('scroll', update, { passive: true, signal });
  window.addEventListener('resize', update, { signal }); update();
  return { destroy() { events.abort(); cancelSettle(); clearPointer(); control.remove(); } };
}
