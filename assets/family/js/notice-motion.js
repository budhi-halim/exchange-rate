const duration = milliseconds => matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : milliseconds;

/** Own animations so detached notices and interrupted list movement release their resources. */
export function createNoticeMotion() {
  const running = new Map();
  function cancel(node) { running.get(node)?.cancel(); running.delete(node); }
  function animate(node, frames, milliseconds, done = () => {}) {
    cancel(node);
    if (!duration(milliseconds) || !node.animate) { done(); return; }
    const animation = node.animate(frames, { duration: milliseconds, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'both' });
    running.set(node, animation);
    animation.onfinish = () => {
      if (running.get(node) !== animation) return;
      running.delete(node); done(); animation.cancel();
    };
  }
  function positions(parent) {
    return new Map([...parent.children].map(node => [node, node.getBoundingClientRect().top]));
  }
  function reflow(before) {
    for (const [node, top] of before) {
      if (!node.isConnected || node.dataset.noticeLeaving) continue;
      cancel(node);
      const offset = top - node.getBoundingClientRect().top;
      if (Math.abs(offset) > .5) animate(node, [{ transform: `translateY(${offset}px)` }, { transform: 'none' }], 220);
    }
  }
  return {
    enter(node) { animate(node, [{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'none' }], 200); },
    remove(node, done, direction = 1) {
      node.dataset.noticeLeaving = 'true'; node.inert = true;
      const style = getComputedStyle(node);
      animate(node, [{ opacity: style.opacity, transform: style.transform }, { opacity: 0, transform: `translateX(${direction * 32}px)` }], 180, () => {
        const before = node.parentElement ? positions(node.parentElement) : new Map();
        node.remove(); done(); reflow(before);
      });
    },
    cancel,
    destroy() { for (const node of running.keys()) cancel(node); },
  };
}
