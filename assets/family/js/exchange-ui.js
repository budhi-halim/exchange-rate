import { createDataViewer } from './table-ui.js';
import { normalizeExchange } from './exchange-logic.js';
import { fetchExchange, EXCHANGE_BASE } from './exchange-service.js';

const columns = [
  { key: 'date', label: 'Date', type: 'date' },
  { key: 'period', label: 'Period', filter: true },
  { key: 'category', label: 'Category', filter: true },
  { key: 'buying', label: 'Actual buying', type: 'number' },
  { key: 'selling', label: 'Actual selling', type: 'number' },
  { key: 'buying_buffered', label: 'Buffered buying', type: 'number' },
  { key: 'selling_buffered', label: 'Buffered selling', type: 'number' },
  { key: 'state', label: 'State', filter: true },
  { key: 'source', label: 'Source timestamp', type: 'date' },
  { key: 'generated', label: 'Generated (UTC)', type: 'date' }
];

for (const link of document.querySelectorAll('a[href="data/today.json"], a[href="data/history.json"]')) {
  link.href = new URL(link.getAttribute('href').split('/').at(-1), EXCHANGE_BASE).href;
}

createDataViewer({
  host: document.querySelector('#viewer'), columns, dateKey: 'date',
  note: 'USD / IDR. Actual and buffered rates from the published Islandsun rate service.',
  load: async signal => {
    const [today, history] = await Promise.all([fetchExchange('today.json', signal), fetchExchange('history.json', signal)]);
    return normalizeExchange(today, history);
  }
});
