import { element, button } from './workspace.js';

/** Checkbox filters keep selections accessible without modifier keys on touch or keyboard. */
export function createMultiSelect(label, onChange) {
  const root = element('details', undefined, 'workspace-multiselect');
  const summary = element('summary');
  const value = element('span', 'All', 'filter-value');
  summary.append(element('span', label), value);
  const panel = element('div', undefined, 'filter-panel');
  const search = element('input'); search.type = 'search'; search.placeholder = 'Find options';
  search.setAttribute('aria-label', `Find ${label.toLowerCase()} options`);
  const choices = element('fieldset'); choices.append(element('legend', `${label} options`, 'isi-sr-only'));
  const list = element('div', undefined, 'filter-options'); choices.append(list);
  const empty = element('p', 'No matching options.', 'isi-muted'); empty.hidden = true;
  let options = [], selected = new Set();
  const clear = button('Clear selection', () => {
    selected.clear(); updateValue(); render(); onChange([]);
  }, 'isi-button isi-button--quiet');
  panel.append(search, clear, choices, empty); root.append(summary, panel);

  function updateValue() {
    value.textContent = selected.size === 0 ? 'All' : selected.size === 1 ? [...selected][0] : `${selected.size} selected`;
    summary.dataset.active = String(selected.size > 0);
  }
  function render() {
    if (!root.open) return;
    const query = search.value.trim().toLocaleLowerCase();
    const fragment = document.createDocumentFragment();
    for (const option of [...new Set([...options, ...selected])]) {
      if (query && !option.toLocaleLowerCase().includes(query)) continue;
      const label = element('label'), checkbox = element('input');
      checkbox.type = 'checkbox'; checkbox.value = option; checkbox.checked = selected.has(option);
      label.append(checkbox, document.createTextNode(option)); fragment.append(label);
    }
    empty.hidden = Boolean(fragment.childElementCount); list.replaceChildren(fragment);
  }
  function position() {
    if (!root.open) return;
    const box = summary.getBoundingClientRect();
    const width = Math.min(320, innerWidth - 24);
    panel.style.width = `${width}px`;
    panel.style.left = `${Math.max(12, Math.min(box.left, innerWidth - width - 12))}px`;
    const below = innerHeight - box.bottom - 12;
    const above = box.top - 12;
    const useBelow = below >= Math.min(300, innerHeight * .45) || below >= above;
    panel.style.maxHeight = `${Math.max(80, Math.min(360, useBelow ? below : above))}px`;
    panel.style.top = useBelow ? `${box.bottom + 4}px` : 'auto';
    panel.style.bottom = useBelow ? 'auto' : `${innerHeight - box.top + 4}px`;
  }
  function close(event) { if (!root.contains(event.target)) root.open = false; }
  function keydown(event) {
    if (event.key === 'Escape' && root.open) { root.open = false; summary.focus(); }
  }
  root.addEventListener('toggle', () => { if (root.open) { render(); position(); } });
  search.addEventListener('input', render);
  list.addEventListener('change', event => {
    const input = event.target;
    if (!input.matches('input[type="checkbox"]')) return;
    if (input.checked) selected.add(input.value); else selected.delete(input.value);
    updateValue(); onChange([...selected]);
  });
  document.addEventListener('pointerdown', close);
  root.addEventListener('keydown', keydown);
  window.addEventListener('resize', position);
  window.addEventListener('scroll', position, { capture: true, passive: true });
  return {
    root,
    setOptions(values) { options = values; render(); },
    setValue(values = []) { selected = new Set(Array.isArray(values) ? values : [values]); search.value = ''; updateValue(); render(); },
    destroy() { document.removeEventListener('pointerdown', close); window.removeEventListener('resize', position); window.removeEventListener('scroll', position, true); root.remove(); }
  };
}
