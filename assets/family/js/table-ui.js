import { element, button, plainText, portalLink } from './workspace.js';
import { readTableState, writeTableState } from './table-logic.js';
import { createMultiSelect } from './multi-select.js';
import { createNotifications } from './notifications.js';

/** A read-only table; a worker owns searching and sorting for each loaded snapshot. */
export function createDataViewer({ host, columns, dateKey, load, note = '', omitDetails = [], detailLabels = {} }) {
  let state = readTableState(location.search, columns);
  let worker = null;
  let controller = null;
  let sequence = 0;
  let loadSequence = 0;
  let debounce = null;
  let destroyed = false;
  let loaded = false;
  let currentRows = [];
  const notificationHost = element('div', undefined, 'isi-toasts');
  document.body.append(notificationHost);
  const notifications = createNotifications(notificationHost);
  document.body.classList.add('isi-data-workspace');
  const toolbar = element('div', undefined, 'workspace-toolbar');
  const searchLabel = element('label', 'Search');
  const search = element('input');
  search.type = 'search'; search.placeholder = 'Search records'; search.value = state.query;
  searchLabel.append(search); toolbar.append(searchLabel);
  const filterSelects = new Map();
  for (const column of columns.filter(column => column.filter)) {
    const select = createMultiSelect(column.label, values => { state.filters[column.key] = values; change(); });
    filterSelects.set(column.key, select); toolbar.append(select.root);
  }
  const dates = new Map();
  if (dateKey) for (const [key, title] of [['from', 'From'], ['to', 'To']]) {
    const label = element('label', title), input = element('input');
    input.type = 'date'; input.value = state[key];
    input.addEventListener('change', () => { state[key] = input.value; change(); });
    dates.set(key, input); label.append(input); toolbar.append(label);
  }
  const actions = element('div', undefined, 'workspace-actions');
  const reset = button('Reset', () => {
    clearTimeout(debounce); notifications.clear(); state = readTableState('', columns); restoreControls(); query();
  });
  const refresh = button('Refresh', refreshData);
  actions.append(reset, refresh);
  const status = element('p', '', 'isi-notice'); status.setAttribute('role', 'status');
  const retry = button('Try again', refreshData); retry.hidden = true;
  const controls = element('details', undefined, 'workspace-columns');
  controls.append(element('summary', 'Columns'));
  const choices = element('div'); controls.append(choices);
  const checkboxes = new Map();
  for (const column of columns) {
    const label = element('label'), checkbox = element('input'); checkbox.type = 'checkbox';
    checkbox.checked = state.visible.includes(column.key);
    checkbox.addEventListener('change', () => {
      state.visible = columns.filter(column => checkboxes.get(column.key).checked).map(column => column.key);
      if (!state.visible.length) {
        checkbox.checked = true; state.visible = [column.key];
        notifications.show('Keep at least one column visible.', { key: 'visible-columns', tone: 'error' });
        return;
      }
      query();
    });
    label.append(checkbox, document.createTextNode(column.label)); choices.append(label); checkboxes.set(column.key, checkbox);
  }
  const region = element('div', undefined, 'isi-table-region'); region.tabIndex = 0; region.setAttribute('aria-label', 'Records');
  region.hidden = true;
  const table = element('table', undefined, 'isi-table isi-table--compact workspace-table');
  const caption = element('caption', 'Records'); caption.className = 'isi-hidden';
  const head = element('thead'), body = element('tbody'); table.append(caption, head, body); region.append(table);
  const footer = element('div', undefined, 'workspace-footer');
  footer.hidden = true;
  const count = element('p', '', 'isi-muted'); count.setAttribute('role', 'status');
  const pages = element('div', undefined, 'workspace-actions');
  const previous = button('Previous', () => { state.page--; query(); });
  const next = button('Next', () => { state.page++; query(); });
  const pageLabel = element('span', '', 'isi-muted');
  const sizeLabel = element('label', 'Rows'); const size = element('select');
  size.setAttribute('aria-label', 'Rows');
  for (const value of [25, 50, 100]) size.append(new Option(String(value), String(value)));
  size.value = state.size;
  size.addEventListener('change', () => { state.size = Number(size.value); change(); });
  sizeLabel.append(size); pages.append(sizeLabel, previous, pageLabel, next); footer.append(count, pages);
  const dialog = element('dialog', undefined, 'workspace-dialog'); dialog.setAttribute('aria-label', 'Record details');
  const dialogHeading = element('div', undefined, 'workspace-title');
  dialogHeading.append(element('h2', 'Record details'), button('Close', () => dialog.close()));
  const details = element('dl'); details.dataset.familySelectable = '';
  dialog.append(dialogHeading, details);
  const tools = element('details', undefined, 'workspace-table-tools');
  tools.append(element('summary', 'Filters and columns'));
  const toolContent = element('div', undefined, 'workspace-tool-content');
  toolContent.append(toolbar, actions, controls);
  if (note) toolContent.append(element('p', note, 'workspace-caption'));
  tools.append(toolContent);
  host.classList.add('workspace-stack', 'workspace-data-viewer');
  host.append(tools, status, retry, region, footer, dialog);

  function documentList(items) {
    const list = element('ul', undefined, 'workspace-document-list');
    for (const item of items) list.append(element('li', item));
    return list;
  }

  function showDetails(row) {
    details.replaceChildren();
    for (const [key, value] of Object.entries(row.details)) {
      const column = columns.find(column => column.source === key || column.key === key);
      const label = detailLabels[key] || column?.label || key.replaceAll('_', ' ');
      const content = element('dd', value || '—');
      if (column?.list) content.replaceChildren(row.lists[column.key]?.length ? documentList(row.lists[column.key]) : document.createTextNode('—'));
      details.append(element('dt', label), content);
    }
    if (row.link) {
      const link = element('a', 'Open in Islandsun portal', 'isi-button'); link.href = row.link;
      link.target = '_blank'; link.rel = 'noopener noreferrer';
      const dd = element('dd'); dd.append(link); details.append(element('dt', 'Source'), dd);
    }
    dialog.showModal();
  }
  function render(result) {
    region.hidden = false; footer.hidden = false;
    currentRows = result.rows;
    state.page = result.page;
    const tr = element('tr'); head.replaceChildren(tr);
    const visible = columns.filter(column => state.visible.includes(column.key));
    for (const column of visible) {
      const th = element('th'); th.scope = 'col';
      th.setAttribute('aria-sort', state.sort === column.key ? (state.direction === 'asc' ? 'ascending' : 'descending') : 'none');
      th.append(button(column.label, () => {
        state.direction = state.sort === column.key && state.direction === 'asc' ? 'desc' : 'asc';
        state.sort = column.key; change();
      }, ''));
      tr.append(th);
    }
    tr.append(element('th', 'Details')); body.replaceChildren();
    for (const row of currentRows) {
      const tr = element('tr');
      for (const column of visible) {
        const cell = element('td', row.values[column.key] || '—', column.list ? 'cell--list' : column.wrap ? 'cell--wrap' : '');
        if (column.list && row.lists[column.key]?.length) cell.replaceChildren(documentList(row.lists[column.key]));
        tr.append(cell);
      }
      const td = element('td', undefined, 'record-action'); const action = button('View', () => showDetails(row));
      action.setAttribute('aria-label', `View record ${row.values[columns.find(column => column.type !== 'date')?.key] || row.id + 1}`);
      td.append(action); tr.append(td); body.append(tr);
    }
    count.textContent = `${result.count.toLocaleString()} of ${result.total.toLocaleString()} records`;
    pageLabel.textContent = `${result.page} / ${result.pages}`;
    previous.disabled = result.page <= 1; next.disabled = result.page >= result.pages;
    status.hidden = result.count > 0;
    status.textContent = result.total ? 'No records match these filters.' : 'No records are available.';
    status.removeAttribute('data-tone'); retry.hidden = true;
    region.setAttribute('aria-busy', 'false');
    history.replaceState(null, '', `${location.pathname}?${writeTableState(state)}${location.hash}`);
  }
  function fail(message) {
    status.hidden = false; status.dataset.tone = 'error'; status.textContent = message;
    retry.hidden = false; region.setAttribute('aria-busy', 'false');
    notifications.show(message, { key: 'data-error', tone: 'error' });
  }
  function query() {
    if (!loaded || destroyed || refresh.disabled) return;
    region.setAttribute('aria-busy', 'true');
    worker.postMessage({ type: 'query', id: ++sequence, state });
  }
  function change() { state.page = 1; query(); }
  function restoreControls() {
    search.value = state.query; size.value = state.size;
    for (const [key, select] of filterSelects) {
      select.setValue(state.filters[key]);
    }
    for (const [key, input] of dates) input.value = state[key];
    for (const [key, input] of checkboxes) input.checked = state.visible.includes(key);
  }
  async function refreshData() {
    const current = ++loadSequence;
    sequence++;
    controller?.abort(); controller = new AbortController();
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(45000)]);
    refresh.disabled = true; retry.hidden = true;
    status.hidden = false; status.removeAttribute('data-tone');
    status.textContent = 'Loading records…';
    notifications.clear();
    try {
      const records = await load(signal);
      if (!Array.isArray(records) || records.some(record => !record || typeof record !== 'object' || Array.isArray(record))) throw new Error('schema');
      const rows = records.map((record, id) => {
        const details = Object.fromEntries(Object.entries(record).filter(([key]) => !omitDetails.includes(key)).map(([key, value]) => [key, plainText(value)]));
        const values = {}, lists = {};
        for (const column of columns) {
          const value = column.value ? column.value(record) : record[column.source || column.key];
          if (column.list) {
            lists[column.key] = (Array.isArray(value) ? value : [value]).map(plainText).filter(Boolean);
            values[column.key] = lists[column.key].join('\n');
          } else values[column.key] = plainText(value);
        }
        let link = null;
        if (record.view) {
          const template = document.createElement('template'); template.innerHTML = String(record.view);
          link = portalLink(template.content.querySelector('a[href]')?.getAttribute('href'));
        }
        return { id, values, lists, details, link, search: Object.values(values).join(' ').toLocaleLowerCase() };
      });
      if (destroyed || current !== loadSequence) return;
      for (const [key, select] of filterSelects) {
        const options = [...new Set(rows.flatMap(row => row.lists[key] || [row.values[key]]).filter(Boolean))].sort((a, b) => a.localeCompare(b));
        select.setOptions(options);
      }
      restoreControls();
      worker?.terminate();
      worker = new Worker(new URL('./table-worker.js', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }) => {
        if (destroyed || data.id !== sequence) return;
        if (data.error) fail('These records could not be displayed. Try loading them again.');
        else render(data);
      };
      worker.onerror = () => fail('The table could not start. Try loading it again.');
      loaded = true; region.setAttribute('aria-busy', 'true');
      worker.postMessage({ type: 'initialize', id: ++sequence, rows, columns: columns.map(({ key, type, numberFormat }) => ({ key, type, numberFormat })), dateKey, state });
    } catch {
      if (!destroyed && current === loadSequence) fail(loaded ? 'Refresh failed. Previously loaded records are still shown.' : 'Data is unavailable or has an unsupported format. Try again later.');
    } finally { if (current === loadSequence) refresh.disabled = false; }
  }
  search.addEventListener('input', () => {
    state.query = search.value; clearTimeout(debounce); debounce = setTimeout(change, 180);
  });
  const restore = () => { clearTimeout(debounce); state = readTableState(location.search, columns); restoreControls(); query(); };
  window.addEventListener('popstate', restore);
  refreshData();
  return { destroy() {
    destroyed = true; sequence++; loadSequence++; clearTimeout(debounce); controller?.abort(); worker?.terminate();
    for (const select of filterSelects.values()) select.destroy();
    notifications.destroy(); notificationHost.remove();
    document.body.classList.remove('isi-data-workspace');
    window.removeEventListener('popstate', restore); host.replaceChildren();
  } };
}

export async function fetchRecords(url, signal, { envelope = false, empty = false } = {}) {
  const response = await fetch(url, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error('unavailable');
  const text = await response.text();
  if (!text.trim() && empty) return [];
  const json = JSON.parse(text);
  return envelope ? json.data : json;
}
