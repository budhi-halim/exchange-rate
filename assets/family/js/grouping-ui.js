import { element, button } from './workspace.js';
import { createGrouping, moveArea, moveGroup, removeGroup, groupingOutput } from './workbook-logic.js';

export function chooseGroups(areas) {
  return new Promise(resolve => {
    const model = createGrouping(areas);
    let dragged = null;
    const dialog = element('dialog', undefined, 'workspace-dialog grouping-dialog');
    dialog.setAttribute('aria-labelledby', 'grouping-title');
    const title = element('h2', 'Group areas'); title.id = 'grouping-title';
    const hint = element('p', 'Drag areas into groups, or use each area’s destination and order controls.', 'isi-muted');
    const contents = element('div', undefined, 'workspace-stack');
    const live = element('p', '', 'isi-muted'); live.setAttribute('role', 'status');
    const actions = element('div', undefined, 'workspace-actions');
    const add = button('Add group', () => { model.groups.push([]); render(); add.focus(); });
    const finish = value => { dialog.close(); dialog.remove(); resolve(value); };
    const confirm = button('Process and download', () => finish(groupingOutput(model)), 'isi-button isi-button--primary');
    actions.append(add, button('Cancel', () => finish(null)), confirm);
    dialog.append(title, hint, contents, live, actions); document.body.append(dialog);
    dialog.addEventListener('cancel', event => { event.preventDefault(); finish(null); });

    function areaCard(name, group, index) {
      const card = element('div', undefined, 'area-card'); card.draggable = true;
      card.append(element('span', name));
      const controls = element('div', undefined, 'area-controls');
      const select = element('select'); select.setAttribute('aria-label', `Destination for ${name}`);
      select.append(new Option('Unassigned', '-1'));
      model.groups.forEach((_, index) => select.append(new Option(`Group ${index + 1}`, String(index))));
      select.value = group;
      select.addEventListener('change', () => {
        const destination = Number(select.value);
        moveArea(model, group, index, destination); render();
        live.textContent = `${name} moved to ${destination < 0 ? 'Unassigned' : `Group ${destination + 1}`}.`;
        const destinationList = destination < 0 ? model.bench : model.groups[destination];
        contents.querySelector(`[data-group="${destination}"] .area-card:nth-child(${destinationList.length}) select`)?.focus();
      });
      const list = group < 0 ? model.bench : model.groups[group];
      const reorder = delta => {
        const next = index + delta;
        moveArea(model, group, index, group, delta > 0 ? next + 1 : next); render();
        contents.querySelector(`[data-group="${group}"] .area-card:nth-child(${next + 1}) select`)?.focus();
        live.textContent = `${name} moved ${delta < 0 ? 'up' : 'down'}.`;
      };
      const up = button('↑', () => reorder(-1)); up.disabled = index === 0; up.setAttribute('aria-label', `Move ${name} up`);
      const down = button('↓', () => reorder(1)); down.disabled = index === list.length - 1; down.setAttribute('aria-label', `Move ${name} down`);
      controls.append(select, up, down); card.append(controls);
      card.addEventListener('dragstart', event => {
        event.stopPropagation(); dragged = { group, index }; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', name);
      });
      card.addEventListener('dragend', () => { dragged = null; });
      card.addEventListener('dragover', event => { if (dragged && !dragged.isGroup) { event.preventDefault(); event.stopPropagation(); } });
      card.addEventListener('drop', event => {
        if (!dragged || dragged.isGroup) return;
        event.preventDefault(); event.stopPropagation();
        const rect = card.getBoundingClientRect();
        moveArea(model, dragged.group, dragged.index, group, index + (event.clientY > rect.top + rect.height / 2 ? 1 : 0));
        dragged = null; render();
      });
      return card;
    }

    function areaList(items, group) {
      const list = element('div', undefined, 'area-list'); list.dataset.group = group;
      items.forEach((name, index) => list.append(areaCard(name, group, index)));
      if (!items.length) list.append(element('p', 'No areas', 'isi-muted'));
      list.addEventListener('dragover', event => { if (dragged && !dragged.isGroup) event.preventDefault(); });
      list.addEventListener('drop', event => {
        if (!dragged || dragged.isGroup) return;
        event.preventDefault(); event.stopPropagation();
        moveArea(model, dragged.group, dragged.index, group); dragged = null; render();
      });
      return list;
    }

    function render() {
      contents.replaceChildren();
      const bench = element('section', undefined, 'group-section'); bench.append(element('h3', 'Unassigned'), areaList(model.bench, -1)); contents.append(bench);
      model.groups.forEach((items, index) => {
        const section = element('section', undefined, 'group-section');
        const header = element('div', undefined, 'workspace-title'); header.draggable = true;
        const controls = element('div', undefined, 'workspace-actions');
        const up = button('↑', () => { moveGroup(model, index, index - 1); render(); add.focus(); }); up.disabled = index === 0; up.setAttribute('aria-label', `Move group ${index + 1} up`);
        const down = button('↓', () => { moveGroup(model, index, index + 1); render(); add.focus(); }); down.disabled = index === model.groups.length - 1; down.setAttribute('aria-label', `Move group ${index + 1} down`);
        controls.append(up, down, button('Remove', () => { removeGroup(model, index); render(); add.focus(); }));
        header.append(element('h3', `Group ${index + 1}`), controls); section.append(header, areaList(items, index)); contents.append(section);
        header.addEventListener('dragstart', event => { dragged = { isGroup: true, index }; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', 'group'); });
        header.addEventListener('dragend', () => { dragged = null; });
        section.addEventListener('dragover', event => { if (dragged?.isGroup) event.preventDefault(); });
        section.addEventListener('drop', event => {
          if (!dragged?.isGroup) return;
          event.preventDefault(); moveGroup(model, dragged.index, index); dragged = null; render();
        });
      });
    }
    render(); dialog.showModal();
  });
}
