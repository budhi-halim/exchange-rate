export const PORTAL_ROOT = 'http://apps.islandsunindonesia.com:81/islandsun/';
export const LOGIN_URL = `${PORTAL_ROOT}index.php/login/`;

export class SessionExpiredError extends Error {
  constructor() { super('Your company session has expired. Sign in and try again.'); this.name = 'SessionExpiredError'; }
}

export async function fetchPortal(url, signal) {
  signal?.throwIfAborted();
  const timeout = AbortSignal.timeout(30000);
  const response = await fetch(url, { credentials: 'include', cache: 'no-store', signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  if ([401, 403].includes(response.status) || /\/login(?:\/|\?|$)/i.test(new URL(response.url || url).pathname)) throw new SessionExpiredError();
  return response;
}

export function isLoginDocument(text) {
  return /<input\b[^>]*\btype\s*=\s*["']?password\b/i.test(text) ||
    /<form\b[^>]*\baction\s*=\s*["'][^"']*\/login\b/i.test(text) ||
    /<form\b[^>]*\bclass\s*=\s*["'][^"']*\blockscreen-credentials\b/i.test(text) ||
    /<title\b[^>]*>[^<]*\bLogin\s*<\/title>/i.test(text);
}

export async function readPortalItems(response) {
  if (!response.ok) throw new Error('The company connection is unavailable.');
  const text = await response.text();
  if (isLoginDocument(text)) throw new SessionExpiredError();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error('The company system returned an unreadable response.'); }
  const items = Array.isArray(data) ? data : data?.data;
  if (!Array.isArray(items)) throw new Error('The company system returned an unexpected response.');
  return items;
}

export function localDate(date = new Date()) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

export function sampleListUrl(from, to) {
  const url = new URL('samplerequest/json', PORTAL_ROOT);
  url.search = new URLSearchParams({ dari: from, sampai: to, fil_status: '', tipe: '' });
  return url.href;
}

export async function checkSession(signal) {
  // Match the original Revenue check: the protected route's final URL determines sign-in state.
  // Signed-in pages may contain password controls or forms under /login/; their HTML is not an auth signal.
  const response = await fetchPortal(PORTAL_ROOT, signal);
  if (!response.ok) throw new Error('The company connection is unavailable.');
  signal?.throwIfAborted();
  const finalUrl = new URL(response.url || PORTAL_ROOT);
  if (finalUrl.origin !== new URL(PORTAL_ROOT).origin || !finalUrl.pathname.startsWith('/islandsun/')) {
    throw new Error('The company system returned an unexpected sign-in response.');
  }
}
