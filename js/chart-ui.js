import { SERIES, valueScale, nearestPoint, formatDate, formatRate, DAY } from './chart-logic.js';

const SVG = 'http://www.w3.org/2000/svg';
function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function createRateChart({ host, caption, announcement }) {
  const events = new AbortController();
  const svg = svgNode('svg', { class: 'rate-chart', role: 'img', tabindex: '0', 'aria-label': 'Exchange rate history. Use left and right arrows to inspect daily rates.', 'aria-describedby': 'chart-help' });
  const tooltip = document.createElement('div');
  tooltip.className = 'chart-inspection'; tooltip.hidden = true;
  host.append(svg, tooltip);
  let points = [], selected = [], geometry = null, inspected = -1, frame = 0, pointerFrame = 0, latestPointer = null;
  let cursor = null, markers = null;
  function hide() {
    cancelAnimationFrame(pointerFrame); pointerFrame = 0;
    tooltip.hidden = true;
    if (cursor) cursor.hidden = true;
    cursor?.setAttribute('visibility', 'hidden');
    markers?.replaceChildren();
    inspected = -1;
  }
  function inspect(index, speak = false) {
    if (!geometry || index < 0 || index >= points.length || !cursor) return;
    inspected = index;
    const point = points[index], x = geometry.x(point.time);
    cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); cursor.setAttribute('visibility', 'visible');
    const heading = document.createElement('strong'); heading.textContent = formatDate(point.date);
    const list = document.createElement('dl');
    markers.replaceChildren();
    for (const series of selected) {
      const label = document.createElement('dt'), value = document.createElement('dd');
      label.textContent = series.label; value.textContent = formatRate(point.values[series.key]);
      label.dataset.color = series.color;
      list.append(label, value);
      if (Number.isFinite(point.values[series.key])) markers.append(svgNode('circle', { cx: x, cy: geometry.y(point.values[series.key]), r: 3.5, class: 'chart-marker', 'data-color': series.color }));
    }
    tooltip.replaceChildren(heading, list); tooltip.hidden = false;
    tooltip.dataset.interactive = String(speak);
    tooltip.dataset.side = x < geometry.width / 2 ? 'right' : 'left';
    if (speak) announcement.textContent = `${formatDate(point.date)}. ${selected.map(series => `${series.label}: ${formatRate(point.values[series.key])}`).join('. ')} IDR per USD.`;
  }
  function draw() {
    frame = 0;
    const { width, height } = host.getBoundingClientRect();
    if (width < 80 || height < 40) { hide(); return; }
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.replaceChildren(svgNode('title', {}, 'Daily USD to IDR exchange rates'), svgNode('desc', {}, 'Each line connects published daily observations. Dashed lines are buffered rates. The vertical scale adapts to the selected series.'));
    hide(); geometry = null;
    const scale = valueScale(points, selected.map(series => series.key), height < 120 ? 2 : height < 200 ? 3 : 5);
    if (!points.length || !selected.length || !scale) {
      svg.append(svgNode('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', class: 'chart-empty' }, !selected.length ? 'Choose a series to display.' : 'No rates in this date range.'));
      caption.textContent = !selected.length ? 'Open Series to choose the rates to compare.' : 'Try another date range or series.';
      return;
    }
    const left = width < 450 ? 53 : 64, right = 18, top = height < 120 ? 18 : 24, bottom = height < 120 ? 22 : 29;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const first = points[0].time, last = points.at(-1).time;
    const x = time => left + (last === first ? plotWidth / 2 : (time - first) / (last - first) * plotWidth);
    const y = value => top + (scale.max - value) / (scale.max - scale.min) * plotHeight;
    geometry = { x, y, width, left, plotWidth, first, last };
    svg.append(svgNode('text', { x: left, y: 13, class: 'chart-axis-title' }, 'IDR per 1 USD'));
    for (const value of scale.ticks) {
      svg.append(svgNode('line', { x1: left, x2: width - right, y1: y(value), y2: y(value), class: 'chart-grid' }));
      svg.append(svgNode('text', { x: left - 8, y: y(value) + 4, 'text-anchor': 'end', class: 'chart-tick' }, formatRate(value)));
    }
    const tickCount = Math.min(points.length, width < 500 ? 3 : 6);
    const dates = new Set(Array.from({ length: tickCount }, (_, index) => Math.round(index * (points.length - 1) / Math.max(1, tickCount - 1))));
    for (const index of dates) svg.append(svgNode('text', { x: x(points[index].time), y: height - 7, 'text-anchor': index === 0 ? 'start' : index === points.length - 1 ? 'end' : 'middle', class: 'chart-tick' }, formatDate(points[index].date, last - first < 365 * DAY)));
    for (const series of selected) {
      let drawing = false, commands = '';
      for (const point of points) {
        const value = point.values[series.key];
        if (!Number.isFinite(value)) { drawing = false; continue; }
        commands += `${drawing ? 'L' : 'M'}${x(point.time).toFixed(2)},${y(value).toFixed(2)} `;
        drawing = true;
      }
      svg.append(svgNode('path', { d: commands, class: 'chart-series', 'data-series': series.key, 'data-color': series.color, 'data-buffered': series.buffered }));
      if (points.length === 1 && Number.isFinite(points[0].values[series.key])) svg.append(svgNode('circle', { cx: x(first), cy: y(points[0].values[series.key]), r: 4, class: 'chart-marker', 'data-color': series.color }));
    }
    cursor = svgNode('line', { y1: top, y2: height - bottom, class: 'chart-cursor', visibility: 'hidden' });
    markers = svgNode('g', { 'aria-hidden': 'true' }); svg.append(cursor, markers);
    caption.textContent = `${formatDate(points[0].date)} – ${formatDate(points.at(-1).date)} · ${points.length} daily observations · dashed = buffered`;
  }
  function redraw() { if (!frame) frame = requestAnimationFrame(draw); }
  function locate(event, speak) {
    if (!geometry) return;
    const bounds = svg.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left - geometry.left) / geometry.plotWidth));
    inspect(nearestPoint(points, geometry.first + fraction * (geometry.last - geometry.first)), speak);
  }
  svg.addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse') return;
    latestPointer = { clientX: event.clientX };
    if (!pointerFrame) pointerFrame = requestAnimationFrame(() => { pointerFrame = 0; locate(latestPointer, false); });
  }, { passive: true, signal: events.signal });
  svg.addEventListener('click', event => locate(event, true), { signal: events.signal });
  svg.addEventListener('pointerleave', event => { if (event.pointerType === 'mouse') hide(); }, { signal: events.signal });
  svg.addEventListener('blur', hide, { signal: events.signal });
  document.addEventListener('pointerdown', event => { if (!host.contains(event.target)) hide(); }, { passive: true, signal: events.signal });
  svg.addEventListener('keydown', event => {
    if (event.key === 'Escape') { hide(); return; }
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || !points.length) return;
    event.preventDefault();
    const step = event.shiftKey ? 7 : 1;
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? points.length - 1 : (inspected < 0 ? points.length - 1 : inspected) + (event.key === 'ArrowLeft' ? -step : step);
    inspect(Math.max(0, Math.min(points.length - 1, next)), true);
  }, { signal: events.signal });
  const resize = new ResizeObserver(redraw); resize.observe(host);
  return {
    update(nextPoints, keys) { points = nextPoints; selected = SERIES.filter(series => keys.includes(series.key)); redraw(); },
    resize: redraw,
    destroy() { events.abort(); resize.disconnect(); cancelAnimationFrame(frame); cancelAnimationFrame(pointerFrame); host.replaceChildren(); }
  };
}
