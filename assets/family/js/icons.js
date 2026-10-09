const PATHS = {
  back: '<path d="M10.5 19.5 3 12m0 0 7.5-7.5M3 12h18"/>',
  home: '<path d="m2.25 12 8.954-8.955a1.125 1.125 0 0 1 1.591 0L21.75 12M4.5 9.75V21h5.25v-6h4.5v6h5.25V9.75"/>',
  plus: '<path d="M12 4.5v15m7.5-7.5h-15"/>',
  close: '<path d="m6 18 12-12M6 6l12 12"/>',
  check: '<path d="m4.5 12.75 6 6 9-13.5"/>',
  reset: '<path d="M3 10a9 9 0 1 1 2.1 8M3 4v6h6"/>',
  print: '<path d="M6 9V3h12v6M6 18H3V9h18v9h-3M6 14h12v7H6zM17 11h1"/>',
  document: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8M8 17h5"/>'
};

export function icon(name) {
  if (!Object.hasOwn(PATHS, name)) throw new Error(`Unknown family icon: ${name}`);
  return `<svg class="isi-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name]}</svg>`;
}
