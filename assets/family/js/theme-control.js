const PATHS = {
  sun: 'M12 3V1m0 22v-2M3 12H1m22 0h-2M5.6 5.6 4.2 4.2m15.6 15.6-1.4-1.4M5.6 18.4l-1.4 1.4M19.8 4.2l-1.4 1.4M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0Z',
  moon: 'M20.9 13.3A9 9 0 0 1 10.7 3.1 9 9 0 1 0 20.9 13.3Z',
};

/** Render the shared theme affordance without owning the page's scheme policy. */
export function renderThemeControl(button, dark) {
  button.dataset.isiThemeControl = '';
  button.dataset.scheme = dark ? 'dark' : 'light';
  button.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  for (const [name, value] of Object.entries({
    class: 'isi-theme-symbol', viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
    'stroke-width': '1.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true',
  })) svg.setAttribute(name, value);
  const path = document.createElementNS(svg.namespaceURI, 'path');
  path.setAttribute('d', dark ? PATHS.moon : PATHS.sun);
  svg.append(path);
  button.replaceChildren(svg);
}
