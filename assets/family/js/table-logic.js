const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });

/** Recognize complete numeric forms; ambiguous/malformed source values remain missing for numeric order. */
export function numericValue(value, format) {
  if (value === '' || value === null || value === undefined) return null;
  let text = String(value).trim();
  if (format === 'grouped') {
    if (/^-?\d{1,3}([.,])\d{3}(?:\1\d{3})*$/.test(text)) text = text.replace(/[.,]/g, '');
    else if (/^-?\d{1,3}(?:\.\d{3})+,\d{1,2}$/.test(text)) text = text.replaceAll('.', '').replace(',', '.');
    else if (/^-?\d{1,3}(?:,\d{3})+\.\d+$/.test(text)) text = text.replaceAll(',', '');
    else if (/^-?\d+,\d{1,2}$/.test(text)) text = text.replace(',', '.');
  }
  if (!/^-?\d+(?:\.\d+)?$/.test(text)) return null;
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}

export function readTableState(search, columns) {
  const params = new URLSearchParams(search);
  const sort = columns.some(column => column.key === params.get('sort')) ? params.get('sort') : columns[0].key;
  const filters = {};
  for (const column of columns.filter(column => column.filter)) {
    const values = [...new Set(params.getAll(`f-${column.key}`).filter(Boolean))];
    if (values.length) filters[column.key] = values;
  }
  const requested = params.get('columns')?.split(',');
  const visible = columns.filter(column => !requested || requested.includes(column.key)).map(column => column.key);
  return {
    query: params.get('q') || '', filters, sort, direction: params.get('dir') === 'asc' ? 'asc' : 'desc',
    page: Math.max(1, Math.floor(Number(params.get('page'))) || 1),
    size: [25, 50, 100].includes(Number(params.get('size'))) ? Number(params.get('size')) : 25,
    from: /^\d{4}-\d{2}-\d{2}$/.test(params.get('from')) ? params.get('from') : '',
    to: /^\d{4}-\d{2}-\d{2}$/.test(params.get('to')) ? params.get('to') : '',
    visible: visible.length ? visible : columns.map(column => column.key)
  };
}

export function writeTableState(state) {
  const params = new URLSearchParams();
  if (state.query) params.set('q', state.query);
  for (const [key, values] of Object.entries(state.filters)) {
    for (const value of Array.isArray(values) ? values : [values]) if (value) params.append(`f-${key}`, value);
  }
  for (const key of ['sort', 'page', 'size', 'from', 'to']) if (state[key]) params.set(key, state[key]);
  params.set('dir', state.direction);
  params.set('columns', state.visible.join(','));
  return params.toString();
}

export function filterTable(rows, columns, state, dateKey) {
  const terms = state.query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const filtered = rows.filter(row => terms.every(term => row.search.includes(term)) &&
    Object.entries(state.filters).every(([key, values]) => {
      if (!values?.length) return true;
      const selected = Array.isArray(values) ? values : [values];
      const items = row.lists?.[key] || [row.values[key]];
      return items.some(item => selected.includes(item));
    }) &&
    (!state.from || (row.values[dateKey] && row.values[dateKey].slice(0, 10) >= state.from)) &&
    (!state.to || (row.values[dateKey] && row.values[dateKey].slice(0, 10) <= state.to)));
  const column = columns.find(column => column.key === state.sort) || columns[0];
  filtered.sort((a, b) => {
    let left = a.values[column.key], right = b.values[column.key];
    if (column.type === 'number') {
      left = numericValue(left, column.numberFormat);
      right = numericValue(right, column.numberFormat);
    }
    if (left === '' || left === null) return right === '' || right === null ? a.id - b.id : 1;
    if (right === '' || right === null) return -1;
    const delta = column.type === 'number' ? left - right : collator.compare(left, right);
    return delta * (state.direction === 'asc' ? 1 : -1) || a.id - b.id;
  });
  const pages = Math.max(1, Math.ceil(filtered.length / state.size));
  const page = Math.min(state.page, pages);
  return { rows: filtered.slice((page - 1) * state.size, page * state.size), count: filtered.length, total: rows.length, page, pages };
}
