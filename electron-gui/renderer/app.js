const mainOrb = document.getElementById('mainOrb');
const subDotsRing = document.getElementById('subDotsRing');
const subDots = document.querySelectorAll('.sub-dot');

let expanded = false;
let dragging = false;
let dragStart = null;
const DRAG_THRESHOLD = 5;

const SUB_DOT_POSITIONS = [
  { angle: -90 },
  { angle: 30 },
  { angle: 150 },
];

const RING_RADIUS = 95;

function positionSubDots(show) {
  subDots.forEach((dot, i) => {
    const angleDeg = SUB_DOT_POSITIONS[i].angle;
    const angleRad = (angleDeg * Math.PI) / 180;
    const x = Math.cos(angleRad) * RING_RADIUS;
    const y = Math.sin(angleRad) * RING_RADIUS;

    dot.style.left = `calc(50% + ${x}px)`;
    dot.style.top = `calc(50% + ${y}px)`;

    if (show) {
      setTimeout(() => dot.classList.add('show'), 80 * i);
    } else {
      dot.classList.remove('show');
    }
  });
}

function doExpand() {
  if (expanded) return;
  expanded = true;
  window.api.expandDot();
  window.api.showBorder();

  mainOrb.classList.remove('idle-pulse');
  subDotsRing.classList.add('visible');
  positionSubDots(true);
}

function doCollapse() {
  if (!expanded) return;
  expanded = false;

  window.api.hideBorder();
  positionSubDots(false);
  subDotsRing.classList.remove('visible');

  setTimeout(() => {
    window.api.collapseDot();
    mainOrb.classList.add('idle-pulse');
  }, 350);
}

/* ── Dragging (mousedown → mousemove → mouseup) ── */
mainOrb.addEventListener('mousedown', (e) => {
  if (expanded) return;
  dragging = false;
  dragStart = { x: e.screenX, y: e.screenY };
});

window.addEventListener('mousemove', (e) => {
  if (!dragStart) return;
  const dx = e.screenX - dragStart.x;
  const dy = e.screenY - dragStart.y;
  if (!dragging && (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD)) {
    dragging = true;
  }
  if (dragging) {
    window.api.dragDot(dx, dy);
    dragStart = { x: e.screenX, y: e.screenY };
  }
});

window.addEventListener('mouseup', () => {
  if (!dragging && dragStart) {
    expanded ? doCollapse() : doExpand();
  }
  dragStart = null;
  setTimeout(() => { dragging = false; }, 10);
});

mainOrb.addEventListener('click', (e) => {
  e.stopPropagation();
});

subDots.forEach((dot) => {
  dot.addEventListener('click', (e) => {
    e.stopPropagation();
    const action = dot.dataset.action;
    doCollapse();

    setTimeout(() => {
      if (action === 'settings') {
        window.api.openSettings();
      } else if (action === 'task') {
        window.api.openTask();
      } else if (action === 'mic') {
        const icon = dot.querySelector('.sub-dot-icon');
        icon.textContent = icon.textContent === '🎙' ? '🔴' : '🎙';
      }
    }, 400);
  });
});

document.getElementById('dotContainer').addEventListener('click', (e) => {
  if (expanded && e.target.id === 'dotContainer') doCollapse();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') window.api.quitApp();
});
