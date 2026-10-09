/** Keep dependent controls unavailable until loading and initial rendering finish. */
export function createReadinessGate({ host, content, logoUrl, load, onReady }) {
  let controller = null;
  let generation = 0;
  let running = false;
  let destroyed = false;
  const panel = document.createElement('section');
  panel.className = 'isi-loading';
  panel.setAttribute('aria-label', 'Application loading');
  const logo = document.createElement('img');
  logo.src = logoUrl;
  logo.alt = '';
  const spinner = document.createElement('span');
  spinner.className = 'isi-spinner';
  spinner.setAttribute('aria-hidden', 'true');
  const title = document.createElement('h2');
  const status = document.createElement('p');
  status.className = 'isi-muted';
  status.setAttribute('role', 'status');
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'isi-button isi-button--primary';
  retry.textContent = 'Try again';
  panel.append(logo, spinner, title, status, retry);
  host.append(panel);

  function loading() {
    content.hidden = true;
    content.inert = true;
    content.setAttribute('aria-busy', 'true');
    panel.hidden = false;
    spinner.hidden = false;
    retry.hidden = true;
    title.textContent = 'Getting things ready';
    status.textContent = 'Loading your workspace. Please wait a moment.';
  }
  async function start() {
    if (destroyed || running) return false;
    running = true;
    const current = ++generation;
    controller = new AbortController();
    loading();
    try {
      const data = await load(controller.signal);
      if (destroyed || current !== generation) return false;
      await onReady(data);
      if (destroyed || current !== generation) return false;
      panel.hidden = true;
      content.hidden = false;
      content.inert = false;
      content.setAttribute('aria-busy', 'false');
      return true;
    } catch {
      if (destroyed || current !== generation) return false;
      spinner.hidden = true;
      title.textContent = 'We couldn’t load your workspace';
      status.textContent = 'Check your connection and try again.';
      retry.hidden = false;
      return false;
    } finally { if (current === generation) running = false; }
  }
  retry.addEventListener('click', start);
  loading();
  return { start, destroy() { destroyed = true; generation++; controller?.abort(); retry.removeEventListener('click', start); panel.remove(); } };
}
