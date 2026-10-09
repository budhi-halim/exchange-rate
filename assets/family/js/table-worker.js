import { filterTable } from './table-logic.js';

let rows = [];
let columns = [];
let dateKey = '';
self.onmessage = ({ data }) => {
  try {
    if (data.type === 'initialize') {
      rows = data.rows;
      columns = data.columns;
      dateKey = data.dateKey;
    }
    self.postMessage({ id: data.id, ...filterTable(rows, columns, data.state, dateKey) });
  } catch { self.postMessage({ id: data.id, error: true }); }
};
