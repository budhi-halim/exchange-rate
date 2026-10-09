import { element } from '../assets/family/js/workspace.js';
import { createDataViewer } from '../assets/family/js/table-ui.js';
import { createMultiSelect } from '../assets/family/js/multi-select.js';
import { createNotifications } from '../assets/family/js/notifications.js';
import { fetchExchange, EXCHANGE_BASE } from '../assets/family/js/exchange-service.js';
import { normalizeExchange } from '../assets/family/js/exchange-logic.js';
import { CATEGORIES, SERIES, DEFAULT_SERIES, dailyRates, rangeBounds, selectRange, formatRate, formatDate } from './chart-logic.js';
import { createRateChart } from './chart-ui.js';

const $ = selector => document.querySelector(selector);
const notifications = createNotifications();
const events = new AbortController();
const chart = createRateChart({ host: $('#chart-host'), caption: $('#chart-caption'), announcement: $('#chart-announcement') });
let snapshot = null, points = [], selected = [...DEFAULT_SERIES], table = null, loading = null, controller = null, destroyed = false;
const columns = [
  { key: 'date', label: 'Date', type: 'date' }, { key: 'period', label: 'Period', filter: true },
  { key: 'category', label: 'Category', filter: true }, { key: 'buying', label: 'Actual buying', type: 'number' },
  { key: 'selling', label: 'Actual selling', type: 'number' }, { key: 'buying_buffered', label: 'Buffered buying', type: 'number' },
  { key: 'selling_buffered', label: 'Buffered selling', type: 'number' }, { key: 'state', label: 'State', filter: true },
  { key: 'source', label: 'Source timestamp', type: 'date' }, { key: 'generated', label: 'Generated (UTC)', type: 'date' }
];

function showCurrent(today) {
  $('#current-rates').replaceChildren();
  $('#current-date').textContent = today?.date ? `${formatDate(today.date)} · ${today.final === true ? 'Final' : today.final === false ? 'Provisional' : 'Published snapshot'}` : 'Current rates are unavailable.';
  for (const category of CATEGORIES) {
    const card = element('section', undefined, 'rate-current-category');
    card.append(element('h2', category.label));
    const values = element('dl');
    for (const buffered of [false, true]) for (const side of ['buying', 'selling']) {
      const pair = element('div');
      pair.append(element('dt', `${buffered ? 'Buffered' : 'Actual'} ${side}`), element('dd', formatRate(today?.[`${category.key}_${side}_rate${buffered ? '_buffered' : ''}`])));
      values.append(pair);
    }
    card.append(values); $('#current-rates').append(card);
  }
}

function updateGraph() {
  const error = $('#range-error');
  try {
    const bounds = rangeBounds(points, $('#chart-duration').value, $('#chart-from').value, $('#chart-to').value);
    error.hidden = true;
    $('#chart-from').removeAttribute('aria-invalid'); $('#chart-to').removeAttribute('aria-invalid');
    const visible = selectRange(points, bounds);
    chart.update(visible, selected);
    $('#chart-legend').replaceChildren(...SERIES.filter(series => selected.includes(series.key)).map(series => {
      const label = element('span', series.label); label.dataset.color = series.color; label.dataset.buffered = series.buffered;
      return label;
    }));
  } catch (failure) {
    error.textContent = failure.message; error.hidden = false;
    $('#chart-from').setAttribute('aria-invalid', 'true'); $('#chart-to').setAttribute('aria-invalid', 'true');
  }
}

const seriesControl = createMultiSelect('Series', labels => {
  selected = SERIES.filter(series => labels.includes(series.label)).map(series => series.key); updateGraph();
}, { emptyLabel: 'None' });
seriesControl.setOptions(SERIES.map(series => series.label));
seriesControl.setValue(SERIES.filter(series => selected.includes(series.key)).map(series => series.label));
$('#series-control').append(seriesControl.root);

async function refreshSnapshot(signal, notify = true) {
  if (loading) return loading;
  const requestController = new AbortController();
  controller = requestController;
  const requestSignal = AbortSignal.any([requestController.signal, AbortSignal.timeout(45000), ...(signal ? [signal] : [])]);
  $('#refresh-rates').disabled = true;
  const status = $('#rate-status'); status.hidden = false; status.removeAttribute('data-tone'); status.textContent = snapshot ? 'Refreshing rates…' : 'Loading rates…';
  loading = (async () => {
    try {
      const [today, history] = await Promise.all([fetchExchange('today.json', requestSignal), fetchExchange('history.json', requestSignal)]);
      const rows = normalizeExchange(today, history), nextPoints = dailyRates(today, history);
      if (destroyed || requestSignal.aborted) throw new DOMException('Request cancelled', 'AbortError');
      snapshot = { today, history, rows }; points = nextPoints;
      showCurrent(today); updateGraph();
      $('#chart-controls').inert = false; status.hidden = true;
      return snapshot;
    } catch (error) {
      const cancelled = requestController.signal.aborted || (signal?.aborted && signal.reason?.name !== 'TimeoutError');
      if (!destroyed && !cancelled) {
        status.textContent = snapshot ? 'Refresh failed. Previously loaded rates are still shown. Use Refresh to try again.' : 'Rates could not be loaded. Check your connection, then choose Refresh.';
        status.dataset.tone = 'error'; status.hidden = false;
        if (notify) notifications.show(status.textContent, { key: 'rates-error', tone: 'error' });
      }
      throw error;
    } finally {
      requestController.abort();
      loading = null;
      if (!destroyed) $('#refresh-rates').disabled = false;
    }
  })();
  return loading;
}

function createTable() {
  if (!snapshot) return;
  table?.destroy();
  let initial = true;
  table = createDataViewer({ host: $('#viewer'), columns, dateKey: 'date',
    note: 'USD / IDR. Current and historical published snapshots. The graph uses one observation per date; the current snapshot takes precedence.',
    load: async signal => {
      if (initial) { initial = false; return snapshot.rows; }
      return (await refreshSnapshot(signal, false)).rows;
    }
  });
  const sources = element('div', undefined, 'rate-source-links');
  for (const [file, title] of [['today.json', 'Current JSON'], ['history.json', 'History JSON']]) {
    const link = element('a', title); link.href = new URL(file, EXCHANGE_BASE).href; link.target = '_blank'; link.rel = 'noopener noreferrer'; sources.append(link);
  }
  $('#viewer .workspace-tool-content').append(sources);
}

function switchView(view, focus = false) {
  for (const type of ['graph', 'table']) {
    const active = type === view;
    $(`#${type}-tab`).setAttribute('aria-selected', String(active));
    $(`#${type}-tab`).tabIndex = active ? 0 : -1;
    $(`#${type}-panel`).hidden = !active;
  }
  if (view === 'table' && !table) createTable();
  if (view === 'graph') chart.resize();
  if (focus) $(`#${view}-tab`).focus();
}
for (const view of ['graph', 'table']) $(`#${view}-tab`).addEventListener('click', () => switchView(view), { signal: events.signal });
$('.rate-view-tabs').addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  switchView(event.key === 'Home' ? 'graph' : event.key === 'End' ? 'table' : $('#graph-tab').getAttribute('aria-selected') === 'true' ? 'table' : 'graph', true);
}, { signal: events.signal });
$('#chart-duration').addEventListener('change', () => {
  $('#custom-range').hidden = $('#chart-duration').value !== 'custom';
  if ($('#chart-duration').value === 'custom' && !$('#chart-from').value) {
    const bounds = rangeBounds(points, '90'); $('#chart-from').value = bounds.from; $('#chart-to').value = bounds.to;
  }
  updateGraph();
}, { signal: events.signal });
for (const id of ['#chart-from', '#chart-to']) $(id).addEventListener('change', updateGraph, { signal: events.signal });
$('#refresh-rates').addEventListener('click', async () => {
  try { await refreshSnapshot(); if (table || !$('#table-panel').hidden) createTable(); } catch { /* The rate status retains the recovery action. */ }
}, { signal: events.signal });
refreshSnapshot().then(() => { if (!$('#table-panel').hidden) createTable(); }).catch(() => {});
window.addEventListener('pagehide', event => {
  if (event.persisted) return;
  destroyed = true; controller?.abort(); events.abort(); chart.destroy(); seriesControl.destroy(); table?.destroy(); notifications.destroy();
});
