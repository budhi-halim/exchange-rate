const CHANNEL_NAME = 'islandsun-launch-v1';
const LAUNCH_PARAMETER = 'isi-launch';
const PUBLIC_ORIGIN = 'https://budhi-halim.github.io';
const APP_ROOTS = new Set(['pricing-form', 'price', 'pl-isi', 'sales-report', 'sales-target', 'general-database', 'exchange-rate', 'isi-ext']);
const normalizePath = pathname => pathname.replace(/\/index\.html$/, '/').replace(/\/$/, '');

function siteOrigin() {
  return location.origin === PUBLIC_ORIGIN || ['127.0.0.1', 'localhost'].includes(location.hostname) ? location.origin : PUBLIC_ORIGIN;
}

/** Confirm a launcher through live memory only; no saved preference survives restart. */
export function initializeNavigation({ root = document.body } = {}) {
  const origin = siteOrigin();
  const sharedUrl = new URL('/isi-share/', origin).href;
  const internalUrl = new URL('/isi/', origin).href;
  let homeUrl = sharedUrl;
  let channel = null;
  let timer = null;
  let requestId = null;
  let destroyed = false;
  const launches = new Map();
  const decorated = new WeakSet();
  const launcher = root.dataset.familyLauncher;
  const isLauncher = ['isi', 'isi-share'].includes(launcher) && normalizePath(location.pathname) === `/${launcher}`;
  const incoming = new URL(location.href);
  const token = incoming.searchParams.get(LAUNCH_PARAMETER);
  if (token) {
    incoming.searchParams.delete(LAUNCH_PARAMETER);
    history.replaceState(history.state, '', incoming);
  }

  function updateLinks() {
    root.querySelectorAll('a[href]').forEach(link => {
      const url = new URL(link.href);
      if (link.hasAttribute('data-family-home') || (url.origin === origin && ['/isi', '/isi-share'].includes(normalizePath(url.pathname)))) {
        link.href = homeUrl;
      }
      if (!isLauncher || decorated.has(link) || !link.matches('.app-link, [data-family-launch]')) return;
      if (url.origin !== location.origin || !APP_ROOTS.has(url.pathname.split('/')[1]) || /\.(json|zip)$/i.test(url.pathname)) return;
      if (!channel) return;
      const launchToken = crypto.randomUUID();
      launches.set(launchToken, normalizePath(url.pathname));
      url.searchParams.set(LAUNCH_PARAMETER, launchToken);
      link.href = url.href;
      decorated.add(link);
    });
  }

  function stopChannel() { channel?.close(); channel = null; clearTimeout(timer); }
  function connect() {
    if (destroyed || channel || typeof BroadcastChannel !== 'function') return;
    try { channel = new BroadcastChannel(CHANNEL_NAME); } catch { return; }
    channel.onmessage = ({ data }) => {
      if (!data || typeof data !== 'object') return;
      if (isLauncher && data.type === 'request' && launches.get(data.token) === data.path && typeof data.requestId === 'string') {
        channel.postMessage({ type: 'confirmed', token: data.token, requestId: data.requestId, launcher });
      } else if (!isLauncher && data.type === 'confirmed' && data.token === token && data.requestId === requestId && ['isi', 'isi-share'].includes(data.launcher)) {
        homeUrl = data.launcher === 'isi' ? internalUrl : sharedUrl;
        updateLinks();
        stopChannel();
      }
    };
  }

  function restore(event) { if (event.persisted && isLauncher) connect(); }
  connect();
  updateLinks();
  if (!isLauncher && token && /^[a-f0-9-]{36}$/i.test(token) && channel) {
    requestId = crypto.randomUUID();
    channel.postMessage({ type: 'request', token, requestId, path: normalizePath(location.pathname) });
    timer = setTimeout(stopChannel, 2000);
  } else if (!isLauncher) stopChannel();
  const observer = new MutationObserver(updateLinks);
  observer.observe(root, { childList: true, subtree: true });
  window.addEventListener('pagehide', stopChannel);
  window.addEventListener('pageshow', restore);
  return { destroy() { destroyed = true; stopChannel(); observer.disconnect(); launches.clear(); window.removeEventListener('pagehide', stopChannel); window.removeEventListener('pageshow', restore); } };
}

if (document.body?.hasAttribute('data-family-app')) initializeNavigation();
