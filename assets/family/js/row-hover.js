/** Re-hit-test the stationary pointer after scrolling, zooming, resizing, or row redraws. */
export function createRowHover({ root, resolveRow, show, hide, move, keyFor = row => row, disabled = () => false, delay = 160 }) {
  const events = new AbortController();
  let pointer = null, row = null, key = null, timer = 0, frame = 0, ratio = devicePixelRatio;
  function dismiss() { clearTimeout(timer); timer = 0; row = null; key = null; hide(); }
  function inspect() {
    frame = 0;
    const hit = pointer && !disabled() ? document.elementFromPoint(pointer.x, pointer.y) : null;
    const next = hit && root.contains(hit) ? resolveRow(hit) : null;
    const nextKey = next ? keyFor(next) : null;
    if (next === row && nextKey === key) { if (next) move?.(pointer); return; }
    dismiss();
    row = next; key = nextKey;
    if (row) timer = setTimeout(() => { timer = 0; if (row?.isConnected && !disabled()) show(row, pointer); }, delay);
  }
  function refresh() { if (!frame) frame = requestAnimationFrame(inspect); }
  function resized() {
    if (pointer && ratio !== devicePixelRatio) {
      pointer = { x: pointer.x * ratio / devicePixelRatio, y: pointer.y * ratio / devicePixelRatio };
    }
    ratio = devicePixelRatio; refresh();
  }
  const signal = events.signal;
  document.addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse') return;
    pointer = { x: event.clientX, y: event.clientY }; refresh();
  }, { passive: true, signal });
  document.addEventListener('pointerout', event => { if (!event.relatedTarget) { pointer = null; dismiss(); } }, { signal });
  document.addEventListener('scroll', refresh, { capture: true, passive: true, signal });
  window.addEventListener('resize', resized, { passive: true, signal });
  window.visualViewport?.addEventListener('resize', resized, { passive: true, signal });
  window.visualViewport?.addEventListener('scroll', refresh, { passive: true, signal });
  window.addEventListener('blur', () => { pointer = null; if (!disabled()) dismiss(); }, { signal });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { pointer = null; if (!disabled()) dismiss(); } }, { signal });
  const owned = node => node.nodeType === Node.ELEMENT_NODE && node.closest('.isi-row-hover,dialog,[data-isi-owned],.isi-portal-notifications');
  const observer = new MutationObserver(records => {
    if (records.some(record => !owned(record.target) && [...record.addedNodes,...record.removedNodes].some(node => !owned(node)))) refresh();
  });
  observer.observe(root, { childList: true, subtree: true });
  const resize = new ResizeObserver(refresh); resize.observe(root);
  return { dismiss, refresh, destroy() { events.abort(); observer.disconnect(); resize.disconnect(); cancelAnimationFrame(frame); dismiss(); } };
}

export function positionHover(popup, pointer) {
  if (!pointer) return;
  const box = popup.getBoundingClientRect();
  const view = window.visualViewport;
  const x = view?.offsetLeft || 0, y = view?.offsetTop || 0;
  const width = view?.width || innerWidth, height = view?.height || innerHeight;
  popup.style.left = `${Math.max(x + 8, Math.min(pointer.x + 12, x + width - box.width - 8))}px`;
  popup.style.top = `${Math.max(y + 8, Math.min(pointer.y + 12, y + height - box.height - 8))}px`;
}

const animations = new WeakMap();
/** Keep the surface noninteractive throughout its 180ms exit and 200ms entrance. */
export function fadeHover(popup, visible, afterHide) {
  const previous = animations.get(popup);
  const opacity = popup.hidden ? 0 : Number(getComputedStyle(popup).opacity);
  previous?.cancel();
  if (visible) popup.hidden = false;
  if (!visible && popup.hidden) { afterHide?.(); return; }
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const animation = popup.animate([{ opacity }, { opacity: visible ? 1 : 0 }], { duration: reduced ? 0 : visible ? 200 : 180, easing: 'ease-out', fill: 'forwards' });
  animations.set(popup, animation);
  animation.onfinish = () => {
    if (animations.get(popup) !== animation) return;
    if (!visible) { popup.hidden = true; afterHide?.(); }
    animation.cancel(); animations.delete(popup);
  };
}
