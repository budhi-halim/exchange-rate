import { createToastCard } from './toast-card.js';
import { createNoticeMotion } from './notice-motion.js';

const TOAST_DURATION = 7000;

/** A bounded transient queue; persistent history remains the caller's responsibility. */
export function createNotifications(host, { visibleOnly = false, toastClass = '' } = {}) {
  const owned = !host;
  if (owned) { host = document.createElement('section'); document.body.append(host); }
  host.classList.add('isi-toasts'); host.setAttribute('aria-label', 'Notifications');
  const entries = new Map(), queue = [], motion = createNoticeMotion();
  let destroyed = false, pumping = false, claiming;
  const available = () => !visibleOnly || (!document.hidden && document.hasFocus());
  function dismiss(key, user = false) {
    const index = queue.findIndex(entry => entry.key === key);
    if (index >= 0) queue.splice(index, 1)[0].onSkip?.();
    if (claiming?.key === key) claiming.cancelled = true;
    const entry = entries.get(key);
    if (!entry || entry.closing) return;
    entry.closing = true; clearTimeout(entry.timer);
    if (user) entry.onDismiss?.();
    motion.remove(entry.node, () => { entries.delete(key); pump(); });
  }
  async function pump() {
    if (pumping || destroyed || !available()) return;
    pumping = true;
    try {
      const limit = innerHeight < 600 ? 1 : 3;
      while (!destroyed && available() && queue.length && entries.size < limit) {
        if (entries.has(queue[0].key)) break;
        const request = queue.shift(); claiming = request;
        let accepted = true;
        try { if (request.beforeShow) accepted = await request.beforeShow(); }
        catch { accepted = false; }
        claiming = null;
        if (!accepted || destroyed || request.cancelled || !available() || (request.isCurrent && !request.isCurrent())) { request.onSkip?.(); continue; }
        const { key, message, tone, onMute } = request;
        const node = createToastCard(message, { tone, href:request.href, customer:request.customer, onDismiss: () => dismiss(key, true), onMute });
        if (toastClass) node.classList.add(toastClass);
        const entry = { ...request, node, timer: null, remaining:TOAST_DURATION, startedAt:null, hovered:false };
        entries.set(key, entry); host.append(node); motion.enter(node);
        const pause = () => {
          if (entry.startedAt !== null) {
            entry.remaining = Math.max(0,entry.remaining-(performance.now()-entry.startedAt));
          }
          entry.startedAt = null; clearTimeout(entry.timer);
        };
        const schedule = () => {
          if (entry.startedAt !== null || entry.closing || document.hidden || entry.hovered || node.contains(document.activeElement)) return;
          if (entry.remaining <= 0) { dismiss(key); return; }
          entry.startedAt = performance.now();
          entry.timer = setTimeout(() => dismiss(key),entry.remaining);
        };
        entry.schedule = schedule; entry.pause = pause;
        node.addEventListener('pointerenter', () => { entry.hovered = true; pause(); });
        node.addEventListener('focusin', pause);
        node.addEventListener('pointerleave', () => { entry.hovered = false; schedule(); });
        node.addEventListener('focusout', () => queueMicrotask(schedule));
        schedule(); request.onShow?.();
      }
    } finally { pumping = false; }
  }
  function show(message, options = {}) {
    const tone = options.tone || 'success', key = options.key || message;
    if (destroyed) return false;
    const existing = entries.get(key) || queue.find(entry => entry.key === key) || (claiming?.key === key ? claiming : null);
    if (existing && existing.message === message && existing.tone === tone && !existing.closing) return true;
    if (existing) dismiss(key);
    if (queue.length >= 100) return false;
    queue.push({ ...options, key, tone, message: String(message).slice(0, 4000) });
    pump(); return true;
  }
  function resume() {
    for (const entry of entries.values()) {
      if (document.hidden) entry.pause(); else entry.schedule();
    }
    pump();
  }
  window.addEventListener('focus', resume); document.addEventListener('visibilitychange', resume);
  return {
    show, dismiss,
    clear() { for (const item of queue.splice(0)) item.onSkip?.(); if (claiming) claiming.cancelled = true; for (const key of entries.keys()) dismiss(key); },
    destroy() {
      destroyed = true; if (claiming) claiming.cancelled = true;
      for (const item of queue.splice(0)) item.onSkip?.();
      window.removeEventListener('focus', resume); document.removeEventListener('visibilitychange', resume);
      motion.destroy();
      for (const entry of entries.values()) { clearTimeout(entry.timer); entry.node.remove(); }
      entries.clear(); if (owned) host.remove();
    },
  };
}
