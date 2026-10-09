import { initializeTheme } from './theme.js';

initializeTheme({ toggle: document.querySelector('#theme-toggle') });

export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function button(text, action, className = 'isi-button') {
  const node = element('button', text, className);
  node.type = 'button';
  if (action) node.addEventListener('click', action);
  return node;
}

/** Parse detached HTML solely to extract text; no source nodes enter the live page. */
export function plainText(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') return JSON.stringify(value);
  const text = String(value);
  if (!/[<&]/.test(text)) return text.trim();
  const template = document.createElement('template');
  template.innerHTML = text;
  template.content.querySelectorAll('script,style,iframe,object,template').forEach(node => node.remove());
  return template.content.textContent.trim();
}

export function portalLink(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && url.hostname === 'apps.islandsunindonesia.com' &&
      ['', '81'].includes(url.port) && !url.username && !url.password && url.pathname.startsWith('/islandsun/') ? url.href : null;
  } catch { return null; }
}
