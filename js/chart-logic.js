export const CATEGORIES = [
  { key: 'e_rate', label: 'E-rate' },
  { key: 'tt_counter', label: 'TT counter' },
  { key: 'bank_notes', label: 'Bank notes' }
];
export const SERIES = CATEGORIES.flatMap((category, index) => ['buying', 'selling'].flatMap((side, sideIndex) => [false, true].map(buffered => ({
  key: `${category.key}_${side}_rate${buffered ? '_buffered' : ''}`,
  category: category.key,
  label: `${category.label} · ${buffered ? 'Buffered' : 'Actual'} ${side}`,
  shortLabel: `${buffered ? 'Buffered' : 'Actual'} ${side}`,
  color: index * 2 + sideIndex,
  buffered
}))));
export const DEFAULT_SERIES = SERIES.filter(series => series.category === 'tt_counter').map(series => series.key);
export const DAY = 86400000;

export function dateTime(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return NaN;
  const timestamp = Date.parse(`${value}T00:00:00Z`);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 10) === value ? timestamp : NaN;
}

export function dailyRates(today, history) {
  if (!Array.isArray(history)) throw new Error('Invalid rate history');
  const days = new Map();
  for (const entry of [...history, ...(today && Object.keys(today).length ? [today] : [])]) {
    if (!entry || !Number.isFinite(dateTime(entry.date))) throw new Error('Invalid rate date');
    const values = Object.fromEntries(SERIES.map(series => {
      const value = entry[series.key];
      if (value !== null && value !== undefined && (typeof value !== 'number' || !Number.isFinite(value) || value < 0)) throw new Error('Invalid rate value');
      return [series.key, value ?? null];
    }));
    days.set(entry.date, { date: entry.date, time: dateTime(entry.date), values });
  }
  return [...days.values()].sort((a, b) => a.time - b.time);
}

export function rangeBounds(points, duration, from, to) {
  if (!points.length) return { from: '', to: '' };
  if (duration === 'custom') {
    if (!Number.isFinite(dateTime(from)) || !Number.isFinite(dateTime(to))) throw new Error('Choose valid start and end dates.');
    if (from > to) throw new Error('The start date must be before the end date.');
    return { from, to };
  }
  const last = points.at(-1);
  const start = duration === 'all' ? points[0].time : Math.max(points[0].time, last.time - (Number(duration) - 1) * DAY);
  return { from: new Date(start).toISOString().slice(0, 10), to: last.date };
}

export function selectRange(points, bounds) {
  return points.filter(point => point.date >= bounds.from && point.date <= bounds.to);
}

export function valueScale(points, selected, count = 5) {
  let minimum = Infinity, maximum = -Infinity;
  for (const point of points) for (const key of selected) {
    const value = point.values[key];
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
  }
  if (!Number.isFinite(minimum)) return null;
  const padding = Math.max((maximum - minimum) * .08, maximum * .002, 1);
  const low = Math.max(0, minimum - padding), high = maximum + padding;
  const rough = (high - low) / Math.max(2, count - 1);
  const power = 10 ** Math.floor(Math.log10(rough));
  const factor = [1, 2, 2.5, 5, 10].find(value => value >= rough / power) || 10;
  const step = power * factor;
  const min = Math.floor(low / step) * step, max = Math.ceil(high / step) * step;
  const ticks = Array.from({ length: Math.round((max - min) / step) + 1 }, (_, index) => min + index * step);
  return { min, max, ticks };
}

export function nearestPoint(points, time) {
  if (!points.length) return -1;
  let low = 0, high = points.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (points[middle].time < time) low = middle + 1;
    else high = middle;
  }
  return low > 0 && time - points[low - 1].time <= points[low].time - time ? low - 1 : low;
}

export const formatRate = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value) : '—';
export const formatDate = (value, short = false) => new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', ...(short ? {} : { year: 'numeric' }), timeZone: 'UTC' }).format(new Date(typeof value === 'number' ? value : dateTime(value)));
