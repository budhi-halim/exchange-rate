const DURATION = { enter: 200, exit: 220, move: 240 };

/** Finish departures before closing gaps and admitting the next queued cards. */
export function createToastMotion(host, onSettled) {
  const animations = new Map(), pending = new Map();
  let busy = false, destroyed = false;

  function cancel(node) {
    const entry = animations.get(node);
    if (!entry) return;
    animations.delete(node);
    entry.animation.cancel(); entry.resolve(false);
  }

  function animate(node, frames, duration) {
    cancel(node);
    if (destroyed) return Promise.resolve(false);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || !node.animate) return Promise.resolve(true);
    return new Promise(resolve => {
      const animation = node.animate(frames, { duration, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'both' });
      animations.set(node, { animation, resolve });
      animation.onfinish = () => {
        if (animations.get(node)?.animation !== animation) return;
        animations.delete(node); animation.cancel(); resolve(true);
      };
    });
  }

  async function drain() {
    if (busy || destroyed) return;
    busy = true;
    try {
      while (pending.size && !destroyed) {
        const batch = [...pending];
        for (const [node] of batch) pending.delete(node);
        await Promise.all(batch.map(([node]) => {
          node.dataset.noticeLeaving = 'true';
          const style = getComputedStyle(node);
          return animate(node, [
            { opacity: style.opacity, transform: style.transform },
            { opacity: 0, transform: 'translateX(48px)' },
          ], DURATION.exit);
        }));
        if (destroyed) return;
        const positions = new Map([...host.children].map(node => [node, node.getBoundingClientRect().top]));
        for (const [node] of batch) node.remove();
        const movement = [];
        for (const node of host.children) {
          cancel(node);
          const offset = positions.get(node) - node.getBoundingClientRect().top;
          if (Math.abs(offset) > .5) movement.push(animate(node, [
            { transform: `translateY(${offset}px)` }, { transform: 'none' },
          ], DURATION.move));
        }
        await Promise.all(movement);
        if (destroyed) return;
        for (const [, done] of batch) done();
      }
    } finally {
      busy = false;
      if (!destroyed) onSettled();
    }
  }

  return {
    get busy() { return busy || pending.size > 0; },
    enter(node) {
      void animate(node, [{ opacity: 0, transform: 'translateY(-8px)' }, { opacity: 1, transform: 'none' }], DURATION.enter);
    },
    remove(node, done) {
      node.inert = true;
      pending.set(node, done);
      queueMicrotask(drain);
    },
    settle() { for (const node of animations.keys()) cancel(node); },
    destroy() {
      destroyed = true; pending.clear();
      for (const node of animations.keys()) cancel(node);
    },
  };
}
