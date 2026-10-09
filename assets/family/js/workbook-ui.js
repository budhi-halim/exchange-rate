import { element, button } from './workspace.js';
import { createReadinessGate } from './readiness.js';
import { readWorkbooks, waitForPython } from './workbook-logic.js';
import { chooseGroups } from './grouping-ui.js';
import { createNotifications } from './notifications.js';

function displayMessage(message, type) {
  const host = document.querySelector('#workbook-message');
  host.replaceChildren(); host.hidden = !message;
  host.className = 'isi-notice'; host.dataset.tone = type === 'error' ? 'error' : type === 'warning' ? 'warning' : 'success';
  // Python writes to an inert template; only sanitized paragraphs and lists enter this live status.
  const template = document.createElement('template'); template.innerHTML = String(message || '');
  const append = (source, target) => {
    for (const node of source.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) target.append(document.createTextNode(node.textContent));
      else if (node.nodeType === Node.ELEMENT_NODE && ['P', 'UL', 'OL', 'LI', 'STRONG', 'B', 'BR'].includes(node.tagName)) {
        const child = element(node.tagName.toLowerCase()); append(node, child); target.append(child);
      } else if (node.nodeType === Node.ELEMENT_NODE && !['SCRIPT', 'STYLE', 'IFRAME', 'OBJECT'].includes(node.tagName)) target.append(document.createTextNode(node.textContent));
    }
  };
  append(template.content, host);
}

function confirmYears() {
  return new Promise(resolve => {
    const dialog = element('dialog', undefined, 'workspace-dialog workspace-stack');
    dialog.setAttribute('aria-label', 'Multiple years');
    const finish = result => { dialog.close(); dialog.remove(); resolve(result); };
    const actions = element('div', undefined, 'workspace-actions');
    actions.append(button('Cancel', () => finish(false)), button('Continue', () => finish(true), 'isi-button isi-button--primary'));
    dialog.append(element('h2', 'Multiple years detected'), element('p', 'The files contain data from more than one year. Continue with these files?'), actions);
    dialog.addEventListener('cancel', event => { event.preventDefault(); finish(false); });
    document.body.append(dialog); dialog.showModal();
  });
}

function download(buffer, title) {
  const blob = new Blob([new Uint8Array(buffer)], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
  const url = URL.createObjectURL(blob), link = element('a');
  const now = new Date();
  link.href = url; link.download = `${title} ${String(now.getDate()).padStart(2, '0')}-${String(now.getMonth() + 1).padStart(2, '0')}-${now.getFullYear()}.xlsx`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function initializeWorkbook({ target = false } = {}) {
  const notifications = createNotifications();
  const message = (text, type) => {
    displayMessage(text, type);
    if (type === 'error') notifications.show('Please check the file selection and the message below it.', { key: 'workbook-error', tone: 'error' });
  };
  const title = target ? 'Sales Target' : 'Sales Report';
  const form = document.querySelector('#extractForm');
  const files = document.querySelector('#files');
  const submit = document.querySelector('#processButton');
  const reset = document.querySelector('#reset-files');
  const processing = document.querySelector('#processing-dialog');
  const processingText = document.querySelector('#processing-text');
  const spinner = processing.querySelector('.isi-spinner');
  const complete = processing.querySelector('.processing-complete');
  let ready = false;
  let running = false;
  const showProcessing = () => { processingText.textContent = 'Processing your files…'; spinner.hidden = false; complete.hidden = true; processing.showModal(); };
  const hideProcessing = () => { if (processing.open) processing.close(); };
  processing.addEventListener('cancel', event => event.preventDefault());
  const gate = createReadinessGate({
    host: document.querySelector('#loading-host'), content: document.querySelector('#workspace-content'),
    logoUrl: 'assets/family/assets/icons/family.svg',
    load: signal => waitForPython(target ? ['prepare_files', 'finalize_files'] : ['process_files'], signal),
    onReady: () => { ready = true; submit.disabled = false; }
  });
  gate.start();
  files.addEventListener('change', () => {
    document.querySelector('#file-count').textContent = files.files.length ? `${files.files.length} file${files.files.length === 1 ? '' : 's'} selected` : 'No files selected';
    displayMessage('', '');
    notifications.clear();
  });
  reset.addEventListener('click', () => {
    if (running) return;
    notifications.clear();
    form.reset(); document.querySelector('#file-count').textContent = 'No files selected'; displayMessage('', ''); files.focus();
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (!ready || running) return;
    if (!files.files.length) { message('Choose at least one XLS or XLSX file.', 'error'); files.focus(); return; }
    running = true; submit.disabled = true; files.disabled = true; reset.disabled = true;
    displayMessage('', '');
    notifications.clear();
    try {
      const infos = await readWorkbooks([...files.files]);
      showProcessing();
      // Let the browser paint the busy state before entering Python's synchronous work.
      await new Promise(resolve => setTimeout(resolve, 30));
      let result;
      if (target) {
        const prepared = await window.prepare_files(infos);
        hideProcessing();
        if (prepared.type === 'error') { message(prepared.message, 'error'); return; }
        const proceed = prepared.multi_year ? await confirmYears() : true;
        if (!proceed) { displayMessage('Processing cancelled.', ''); return; }
        const groups = await chooseGroups(prepared.areas);
        if (groups === null) { displayMessage('Processing cancelled.', ''); return; }
        showProcessing(); await new Promise(resolve => setTimeout(resolve, 30));
        result = await window.finalize_files(infos, groups, proceed);
      } else result = await window.process_files(infos);
      if (!result || typeof result !== 'object') throw new Error('result');
      if (result.type !== 'error') {
        spinner.hidden = true; complete.hidden = false; processingText.textContent = 'Processing complete.';
        if (result.buffer) download(result.buffer, title);
        await new Promise(resolve => setTimeout(resolve, 1500));
      }
      message(result.message || 'Processing complete.', result.type || 'success');
    } catch (error) {
      message(error.message === 'file-type' ? 'Choose XLS or XLSX files only.' : 'The files could not be processed. Check the input format and try again.', 'error');
    } finally {
      hideProcessing(); running = false; submit.disabled = false; files.disabled = false; reset.disabled = false; submit.focus();
    }
  });
}
