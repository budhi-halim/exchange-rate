const PATHS = {
  success: 'm4.5 12.75 6 6 9-13.5',
  warning: 'M12 8v4.5l3 1.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  error: 'm9 9 6 6m0-6-6 6M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  close: 'm6 18 12-12M6 6l12 12',
  mute: 'm3 3 18 18M6 6c-1 2-1 5-1 8l-2 3h14M9 20a3 3 0 0 0 6 0M10 3a6 6 0 0 1 8 6v3',
};

function mark(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  for (const [key, value] of Object.entries({ viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' })) svg.setAttribute(key, value);
  const path = document.createElementNS(svg.namespaceURI, 'path'); path.setAttribute('d', PATHS[name]); svg.append(path); return svg;
}

export function createToastCard(message, { tone = 'success', onDismiss, onMute } = {}) {
  const node = document.createElement('div'); node.className = 'isi-toast'; node.dataset.tone = tone;
  node.setAttribute('role', tone === 'error' ? 'alert' : 'status'); node.setAttribute('aria-atomic', 'true');
  const symbol = mark(Object.hasOwn(PATHS, tone) ? tone : 'success'); symbol.classList.add('isi-toast-mark');
  const text = document.createElement('span'); text.className = 'isi-toast-message'; text.textContent = message;
  const actions = document.createElement('div'); actions.className = 'isi-toast-actions';
  function action(name, label, callback) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'isi-toast-button';
    button.setAttribute('aria-label', label); button.append(mark(name)); button.addEventListener('click', callback); actions.append(button);
  }
  if (onMute) action('mute', 'Never remind me again', onMute);
  action('close', 'Dismiss notification', onDismiss);
  node.append(symbol, text, actions); return node;
}
