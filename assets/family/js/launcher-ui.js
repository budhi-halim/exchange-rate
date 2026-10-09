import { element } from './workspace.js';

const descriptions = {
  'Product List': 'Products, prices and production history',
  'Price Calculator': 'Pricing across product groups',
  'Pricing Form': 'Company pricing request documents',
  'Exchange Rate': 'Actual and buffered currency rates',
  'Sample Request': 'Sample requests and follow-up status',
  'Stock Request': 'Stock requests and quantities',
  'Sales Order': 'Sales orders and customer references',
  'Sales Report': 'Sales workbooks from your source files',
  'Sales Target': 'Sales targets and area grouping',
  'Technical Information': 'Requested documents and technical information',
  'Browser Extension': 'Portal Workspace, revenue extraction and feedback'
};
const icons = {
  'Product List': 'M4 6h16M4 12h16M4 18h16',
  'Price Calculator': 'M8 3h8a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2ZM9 7h6M9 11h.01M15 11h.01M9 15h.01M15 15h.01',
  'Pricing Form': 'M14 3H6v18h12V7l-4-4Zm0 0v5h4M9 12h6M9 16h6',
  'Exchange Rate': 'M4 8h16m-4-4 4 4-4 4M20 16H4m4-4-4 4 4 4',
  'Sales Report': 'M4 20h16M7 16v-4m5 4V5m5 11V9',
  'Sales Target': 'M21 12a9 9 0 1 1-9-9m0 4a5 5 0 1 0 5 5m-5 0 8-8'
};
const grid = document.querySelector('#appGrid');
const status = document.querySelector('#launcher-status');
const retry = document.querySelector('#launcher-retry');
let loading = false;

async function load() {
  if (loading) return;
  loading = true; retry.hidden = true; status.hidden = false; status.textContent = 'Loading apps…';
  try {
    const response = await fetch('data/app_list.json', { cache: 'no-store', signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error('unavailable');
    const apps = await response.json();
    if (!Array.isArray(apps)) throw new Error('schema');
    const fragment = document.createDocumentFragment();
    for (const app of apps) {
      const url = new URL(app.url, location.href);
      if (app.name === 'Technical Information' && ['localhost', '127.0.0.1'].includes(location.hostname) && url.hostname === 'budhi-halim.github.io') {
        url.host = location.host; url.protocol = location.protocol;
      }
      if (!['http:', 'https:'].includes(url.protocol) || typeof app.name !== 'string') throw new Error('schema');
      const link = element('a', undefined, 'isi-card workspace-card-link app-link');
      link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      for (const [key, value] of Object.entries({ viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'aria-hidden': 'true' })) svg.setAttribute(key, value);
      const path = document.createElementNS(svg.namespaceURI, 'path'); path.setAttribute('d', icons[app.name] || 'M6 3h12v18H6V3Zm3 5h6m-6 4h6m-6 4h4');
      path.setAttribute('stroke-linecap', 'round'); path.setAttribute('stroke-linejoin', 'round'); svg.append(path);
      const icon = element('span', undefined, 'launcher-icon'); icon.append(svg);
      const arrow = element('span', '↗', 'card-arrow'); arrow.setAttribute('aria-hidden', 'true');
      link.append(icon, element('h2', app.name), element('p', descriptions[app.name] || '', 'isi-muted'), arrow); fragment.append(link);
    }
    grid.replaceChildren(fragment); status.hidden = apps.length > 0; status.textContent = 'No apps are available.';
  } catch { status.textContent = 'Apps could not be loaded. Check your connection and try again.'; retry.hidden = false; }
  finally { loading = false; }
}
retry.addEventListener('click', load);
load();
