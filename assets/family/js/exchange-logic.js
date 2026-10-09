const categories = [['e_rate', 'E-rate'], ['tt_counter', 'TT counter'], ['bank_notes', 'Bank notes']];

export function normalizeExchange(today, history) {
  if (!Array.isArray(history) || (today !== null && (typeof today !== 'object' || Array.isArray(today)))) throw new Error('schema');
  const records = [];
  const entries = history.map(entry => ({ ...entry, period: 'History' }));
  if (today && Object.keys(today).length) entries.unshift({ ...today, period: 'Current' });
  for (const entry of entries) {
    if (!entry || !/^\d{4}-\d{2}-\d{2}$/.test(entry.date)) throw new Error('schema');
    for (const [key, category] of categories) {
      const row = {
        date: entry.date, period: entry.period, category,
        state: entry.period === 'History' ? 'Final' : entry.final === true ? 'Final' : entry.final === false ? 'Provisional' : 'Unknown',
        source: (entry.selected_source_ts || entry.source_ts)?.[key] || '',
        generated: entry.generated_ts_utc || '', note: entry.note || ''
      };
      for (const side of ['buying', 'selling']) for (const suffix of ['', '_buffered']) {
        const value = entry[`${key}_${side}_rate${suffix}`];
        if (value !== null && value !== undefined && (typeof value !== 'number' || !Number.isFinite(value))) throw new Error('schema');
        row[`${side}${suffix}`] = value ?? '';
      }
      records.push(row);
    }
  }
  return records;
}
