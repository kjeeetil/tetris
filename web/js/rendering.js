import {
  BOARD_BG,
  DEFAULT_CELL,
  DEFAULT_PREVIEW_CELL,
  GRID_LINE,
  HEIGHT,
  MAX_LOG_LINES,
  MIN_CELL_SIZE,
  PREVIEW_STROKE,
  SCALE_STEPS,
  SHAPE_COLORS,
  WIDTH,
} from './constants.js';
import { SHAPES } from './engine.js';

function hexToRgb(hex) {
  if (!hex) return null;
  const normalized = hex.replace('#', '');
  const expanded = normalized.length === 3 ? normalized.split('').map((c) => c + c).join('') : normalized;
  if (expanded.length !== 6) return null;
  const r = parseInt(expanded.slice(0, 2), 16);
  const g = parseInt(expanded.slice(2, 4), 16);
  const b = parseInt(expanded.slice(4, 6), 16);
  if ([r, g, b].some((value) => Number.isNaN(value))) return null;
  return [r, g, b];
}

function rgbToHex(rgb) {
  if (!rgb) return null;
  return (
    '#'
    + rgb
      .map((value) => {
        const clamped = Math.max(0, Math.min(255, Math.round(value)));
        return clamped.toString(16).padStart(2, '0');
      })
      .join('')
  );
}

function mixColor(hex, mixHex, weight = 0.5) {
  const base = hexToRgb(hex);
  const mix = hexToRgb(mixHex);
  if (!base || !mix) return hex;
  const w = Math.max(0, Math.min(1, weight));
  const blended = base.map((value, index) => value * (1 - w) + mix[index] * w);
  return rgbToHex(blended) || hex;
}

function lightenHex(hex, amount = 0.25) {
  return mixColor(hex, '#ffffff', amount);
}

function darkenHex(hex, amount = 0.25) {
  return mixColor(hex, '#000000', amount);
}

function paintBlock(ctx, color, x, y, size, options = {}) {
  const { shadow = true, stroke = GRID_LINE } = options;
  if (!ctx) return;
  const gradient = ctx.createLinearGradient(x, y, x + size, y + size);
  gradient.addColorStop(0, lightenHex(color, 0.35));
  gradient.addColorStop(0.5, color);
  gradient.addColorStop(1, darkenHex(color, 0.2));
  ctx.save();
  ctx.fillStyle = gradient;
  if (shadow) {
    ctx.shadowColor = lightenHex(color, 0.35);
    ctx.shadowBlur = 8;
  }
  ctx.fillRect(x, y, size, size);
  if (shadow) {
    ctx.shadowBlur = 0;
  }
  ctx.strokeStyle = stroke;
  ctx.strokeRect(x, y, size, size);
  ctx.restore();
}

export function createRenderer({ canvas, preview, diagnostics, scoreEl, levelEl }) {
  const ctx = canvas ? canvas.getContext('2d') : null;
  const pctx = preview ? preview.getContext('2d') : null;
  let CELL = DEFAULT_CELL;
  let PREV_CELL = DEFAULT_PREVIEW_CELL;
  let skipRender = () => false;
  let stateProvider = null;
  let resizeRaf = null;

  function setSkipCallback(fn) {
    skipRender = typeof fn === 'function' ? fn : () => false;
  }

  function setStateProvider(provider) {
    stateProvider = typeof provider === 'function' ? provider : null;
  }

  function log(message) {
    if (!diagnostics) return;
    const entry = document.createElement('div');
    const now = new Date().toLocaleTimeString();
    entry.textContent = `[${now}] ${message}`;
    diagnostics.prepend(entry);
    while (diagnostics.childElementCount > MAX_LOG_LINES) {
      diagnostics.removeChild(diagnostics.lastChild);
    }
  }

  function computeEffectiveBoardWidth() {
    if (!canvas) return DEFAULT_CELL * WIDTH;
    const parent = canvas.parentElement;
    const viewportWidth = Math.max(
      0,
      Math.min(
        window.innerWidth || 0,
        document.documentElement ? document.documentElement.clientWidth : 0,
      ),
    );
    let effectiveWidth = parent && parent.clientWidth ? parent.clientWidth : 0;
    if (viewportWidth > 0) {
      effectiveWidth = effectiveWidth > 0 ? Math.min(effectiveWidth, viewportWidth) : viewportWidth;
    }
    if (!effectiveWidth || !Number.isFinite(effectiveWidth)) {
      return DEFAULT_CELL * WIDTH;
    }
    const maxBoardWidth = SCALE_STEPS[0] * WIDTH;
    return Math.max(MIN_CELL_SIZE * WIDTH, Math.min(maxBoardWidth, effectiveWidth));
  }

  function snapCellSize(rawCell) {
    if (!Number.isFinite(rawCell) || rawCell <= 0) {
      return CELL;
    }
    for (const step of SCALE_STEPS) {
      if (rawCell >= step) {
        return step;
      }
    }
    const fallback = Math.max(MIN_CELL_SIZE, Math.floor(rawCell));
    return fallback > 0 ? fallback : MIN_CELL_SIZE;
  }

  function applyCanvasScale() {
    if (!canvas || !preview) {
      return false;
    }
    const effectiveWidth = computeEffectiveBoardWidth();
    const rawCell = Math.max(MIN_CELL_SIZE, Math.floor(effectiveWidth / WIDTH));
    const nextCell = snapCellSize(rawCell);
    if (!Number.isFinite(nextCell) || nextCell <= 0) {
      return false;
    }
    const boardChanged = nextCell !== CELL;
    CELL = nextCell;
    PREV_CELL = Math.max(MIN_CELL_SIZE - 2, Math.round(CELL * 0.9));
    const boardWidth = CELL * WIDTH;
    const boardHeight = CELL * HEIGHT;
    if (canvas.width !== boardWidth) {
      canvas.width = boardWidth;
    }
    if (canvas.height !== boardHeight) {
      canvas.height = boardHeight;
    }
    canvas.style.width = `${boardWidth}px`;
    canvas.style.height = `${boardHeight}px`;

    const previewExtent = PREV_CELL * 4 + Math.max(PREV_CELL, 24);
    if (preview.width !== previewExtent) {
      preview.width = previewExtent;
    }
    if (preview.height !== previewExtent) {
      preview.height = previewExtent;
    }
    preview.style.width = `${previewExtent}px`;
    preview.style.height = `${previewExtent}px`;

    return boardChanged;
  }

  function draw(grid, active) {
    if (!ctx || !grid) return;
    if (skipRender()) return;
    ctx.fillStyle = BOARD_BG;
    ctx.fillRect(0, 0, WIDTH * CELL, HEIGHT * CELL);
    ctx.lineWidth = 1.2;
    for (let r = 0; r < HEIGHT; r += 1) {
      for (let c = 0; c < WIDTH; c += 1) {
        const value = grid[r][c];
        const x = c * CELL;
        const y = r * CELL;
        if (value) {
          const color = SHAPE_COLORS[value] || '#6c7dd9';
          paintBlock(ctx, color, x, y, CELL, { shadow: false });
        } else {
          ctx.strokeStyle = GRID_LINE;
          ctx.strokeRect(x, y, CELL, CELL);
        }
      }
    }
    if (active) {
      const color = SHAPE_COLORS[active.shape] || '#6c7dd9';
      for (const [r, c] of active.blocks()) {
        paintBlock(ctx, color, c * CELL, r * CELL, CELL, { shadow: true });
      }
    }
  }

  function drawNext(shape) {
    if (!pctx) return;
    if (skipRender()) return;
    const width = preview ? preview.width : 0;
    const height = preview ? preview.height : 0;
    pctx.clearRect(0, 0, width, height);
    if (!shape) return;
    const state = SHAPES[shape][0];
    let minR = Infinity;
    let minC = Infinity;
    let maxR = -Infinity;
    let maxC = -Infinity;
    for (const [r, c] of state) {
      minR = Math.min(minR, r);
      minC = Math.min(minC, c);
      maxR = Math.max(maxR, r);
      maxC = Math.max(maxC, c);
    }
    const w = (maxC - minC + 1) * PREV_CELL;
    const h = (maxR - minR + 1) * PREV_CELL;
    const offX = Math.floor(((preview ? preview.width : w) - w) / 2);
    const offY = Math.floor(((preview ? preview.height : h) - h) / 2);
    for (const [r, c] of state) {
      const x = offX + (c - minC) * PREV_CELL;
      const y = offY + (r - minR) * PREV_CELL;
      const color = SHAPE_COLORS[shape] || '#6c7dd9';
      paintBlock(pctx, color, x, y, PREV_CELL, { shadow: false, stroke: PREVIEW_STROKE });
    }
  }

  function updateScore(score, force = false) {
    if (!scoreEl) return;
    if (!force && skipRender()) return;
    scoreEl.textContent = `Score: ${Number(score || 0).toLocaleString()}`;
  }

  function updateLevel(level, force = false) {
    if (!levelEl) return;
    if (!force && skipRender()) return;
    levelEl.textContent = `Level: ${Number(level || 0).toLocaleString()}`;
  }

  function drawCurrentState() {
    if (!stateProvider) return;
    const snapshot = stateProvider();
    if (!snapshot) return;
    draw(snapshot.grid, snapshot.active);
    drawNext(snapshot.next);
  }

  function scheduleScaleUpdate() {
    if (resizeRaf !== null) {
      cancelAnimationFrame(resizeRaf);
    }
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = null;
      if (applyCanvasScale()) {
        drawCurrentState();
      }
    });
  }

  function registerResizeListeners() {
    window.addEventListener('resize', scheduleScaleUpdate, { passive: true });
    if (window.visualViewport && typeof window.visualViewport.addEventListener === 'function') {
      window.visualViewport.addEventListener('resize', scheduleScaleUpdate, { passive: true });
    }
  }

  return {
    applyCanvasScale,
    draw,
    drawCurrentState,
    drawNext,
    log,
    registerResizeListeners,
    scheduleScaleUpdate,
    setSkipCallback,
    setStateProvider,
    updateLevel,
    updateScore,
  };
}
