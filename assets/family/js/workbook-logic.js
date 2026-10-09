export async function readWorkbooks(files) {
  const infos = [];
  for (const file of files) {
    if (!/\.xlsx?$/i.test(file.name)) throw new Error('file-type');
    infos.push({ name: file.name, data: Array.from(new Uint8Array(await file.arrayBuffer())) });
  }
  return infos;
}

export function waitForPython(names, signal) {
  return new Promise((resolve, reject) => {
    let interval;
    let timeout;
    const cleanup = () => { clearInterval(interval); clearTimeout(timeout); signal.removeEventListener('abort', aborted); window.removeEventListener('py:all-done', check); };
    const aborted = () => { cleanup(); reject(new Error('cancelled')); };
    const check = () => { if (names.every(name => typeof window[name] === 'function')) { cleanup(); resolve(); } };
    if (signal.aborted) { aborted(); return; }
    signal.addEventListener('abort', aborted, { once: true });
    window.addEventListener('py:all-done', check);
    interval = setInterval(check, 100);
    timeout = setTimeout(() => { cleanup(); reject(new Error('setup-timeout')); }, 90000);
    check();
  });
}

/** Single assignment, ordered areas, and ordered groups match the workbook interface. */
export function createGrouping(areas) {
  return { bench: [...areas], groups: [[], []] };
}

export function moveArea(model, from, index, to, position) {
  const source = from < 0 ? model.bench : model.groups[from];
  const target = to < 0 ? model.bench : model.groups[to];
  if (!source || !target || index < 0 || index >= source.length) return;
  const [name] = source.splice(index, 1);
  const adjusted = source === target && position > index ? position - 1 : position;
  target.splice(Math.max(0, Math.min(adjusted ?? target.length, target.length)), 0, name);
}

export function removeGroup(model, index) {
  const [group] = model.groups.splice(index, 1);
  if (group) model.bench.push(...group);
}

export function moveGroup(model, from, to) {
  if (from < 0 || from >= model.groups.length || to < 0 || to >= model.groups.length) return;
  const [group] = model.groups.splice(from, 1); model.groups.splice(to, 0, group);
}

export function groupingOutput(model) { return model.groups.filter(group => group.length).map(group => [...group]); }
