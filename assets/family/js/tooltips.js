export function initializeTooltips(root = document) {
  const cleanups = [];
  let active = null;
  function close() {
    if (!active) return;
    active.tip.remove();
    if (active.previous) active.trigger.setAttribute('aria-describedby', active.previous);
    else active.trigger.removeAttribute('aria-describedby');
    active = null;
  }
  function position() {
    if (!active) return;
    const box = active.trigger.getBoundingClientRect();
    const tip = active.tip.getBoundingClientRect();
    active.tip.style.left = `${Math.max(12, Math.min(innerWidth - tip.width - 12, box.left + (box.width - tip.width) / 2))}px`;
    active.tip.style.top = `${Math.max(12, Math.min(innerHeight - tip.height - 12, box.top >= tip.height + 20 ? box.top - tip.height - 8 : box.bottom + 8))}px`;
  }
  for (const trigger of root.querySelectorAll('[data-tooltip]')) {
    let openOnPointerDown = false;
    function open() {
      if (active?.trigger === trigger) return;
      close();
      const tip = document.createElement('div');
      tip.id = `isi-tooltip-${crypto.randomUUID()}`;
      tip.className = 'isi-ui isi-tooltip';
      tip.setAttribute('role', 'tooltip');
      tip.textContent = trigger.dataset.tooltip;
      const previous = trigger.getAttribute('aria-describedby');
      trigger.setAttribute('aria-describedby', [previous, tip.id].filter(Boolean).join(' '));
      document.body.append(tip);
      active = { tip, trigger, previous };
      position();
    }
    function leave() { if (document.activeElement !== trigger && active?.trigger === trigger) close(); }
    function blur() { if (active?.trigger === trigger) close(); }
    function enter(event) { if (event.pointerType !== 'touch') open(); }
    function pointerDown() { openOnPointerDown = active?.trigger === trigger; }
    function click(event) { if (event.detail === 0 ? active?.trigger === trigger : openOnPointerDown) close(); else open(); }
    for (const [event, listener] of [['pointerenter', enter], ['pointerdown', pointerDown], ['pointerleave', leave], ['focus', open], ['blur', blur], ['click', click]]) {
      trigger.addEventListener(event, listener);
      cleanups.push(() => trigger.removeEventListener(event, listener));
    }
  }
  const keydown = event => { if (event.key === 'Escape') close(); };
  const outside = event => { if (active && !active.trigger.contains(event.target)) close(); };
  const onScroll = () => { if (active?.trigger === document.activeElement) position(); else close(); };
  document.addEventListener('keydown', keydown);
  document.addEventListener('pointerdown', outside);
  window.addEventListener('resize', position);
  window.addEventListener('scroll', onScroll, true);
  return { destroy() { close(); cleanups.forEach(cleanup => cleanup()); document.removeEventListener('keydown', keydown); document.removeEventListener('pointerdown', outside); window.removeEventListener('resize', position); window.removeEventListener('scroll', onScroll, true); } };
}
