import { clearRows, canMove, emptyGrid, gravityForLevel, lock, Piece, SHAPES, shuffle } from './engine.js';

const BLOCK_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'PageUp', 'PageDown', 'Home', 'End']);

export function createGame(dom, renderer) {
  const { canvas, toggleBtn, resetBtn, speedSlider, speedDisplay } = dom;
  const log = renderer.log;
  const draw = renderer.draw;
  const drawNext = renderer.drawNext;

  const state = {
    grid: emptyGrid(),
    active: null,
    next: null,
    score: 0,
    level: 0,
    pieces: 0,
    running: false,
    paused: false,
    last: 0,
    acc: 0,
    gravity: gravityForLevel(0),
    raf: null,
    wdAcc: 0,
    lastSig: '',
    renderEvery: 1,
    renderCounter: 0,
  };

  let speedMult = 1;
  let bag = [];
  const shapes = Object.keys(SHAPES);

  let trainingHooks = {
    aiStep: () => {},
    isEnabled: () => false,
    shouldSkipRendering: () => false,
  };
  let gameOverHandler = null;

  function updateScore(force = false) {
    renderer.updateScore(state.score, force);
  }

  function updateLevel(force = false) {
    renderer.updateLevel(state.level, force);
  }

  function recordClear() {
    // Legacy stub retained for compatibility.
  }

  function drawFromBag() {
    if (bag.length === 0) {
      bag = shuffle(shapes.slice());
    }
    return bag.pop();
  }

  function spawn() {
    const nextShape = state.next || drawFromBag();
    state.active = new Piece(nextShape);
    state.next = drawFromBag();
    drawNext(state.next);
    updateScore();
    updateLevel();
  }

  function triggerGameOver() {
    if (typeof gameOverHandler === 'function') {
      try {
        gameOverHandler();
        return;
      } catch (err) {
        log(`Game over handler failed: ${err && err.message ? err.message : err}`);
      }
    }
    log('Game over (fallback). Resetting.');
    Object.assign(state, { grid: emptyGrid(), active: null, next: null, score: 0, level: 0, pieces: 0 });
    state.gravity = gravityForLevel(0);
    updateLevel();
    updateScore();
    spawn();
  }

  function updateRenderDecimation() {
    state.renderEvery = 1;
    state.renderCounter = 0;
  }

  function scheduleNext() {
    let delay = 1000 / 60;
    if (trainingHooks.shouldSkipRendering()) {
      delay = 0;
    }
    state.raf = setTimeout(() => {
      state.raf = null;
      tick(performance.now());
    }, delay);
  }

  function tick(ts) {
    try {
      if (!state.running) {
        return;
      }
      if (!state.last) {
        state.last = ts;
      }
      const dt = ts - state.last;
      state.last = ts;
      let effDt = dt * speedMult;
      if (trainingHooks.shouldSkipRendering()) {
        effDt = dt;
      }
      const sig = state.active
        ? `${state.active.shape}:${state.active.rot}:${state.active.row}:${state.active.col}:${state.score}:${state.pieces}`
        : `none:${state.score}:${state.pieces}`;
      if (sig === state.lastSig) {
        state.wdAcc += dt;
      } else {
        state.wdAcc = 0;
        state.lastSig = sig;
      }
      if (state.wdAcc > 2000 && !state.paused) {
        log('Watchdog: no progress for 2s -> game over');
        triggerGameOver();
        state.wdAcc = 0;
      }
      if (!state.paused && !state.active) {
        spawn();
        if (!canMove(state.grid, state.active, 0, 0)) {
          log('Tick: spawned into block -> game over');
          triggerGameOver();
          return;
        }
      }
      if (!state.paused && state.active) {
        if (!canMove(state.grid, state.active, 0, 0)) {
          log('Tick: spawn blocked -> game over');
          triggerGameOver();
          return;
        }
        if (trainingHooks.isEnabled()) {
          trainingHooks.aiStep(effDt);
        } else {
          state.acc += effDt;
          while (state.acc >= state.gravity) {
            state.acc -= state.gravity;
            if (canMove(state.grid, state.active, 0, 1)) {
              state.active.move(0, 1);
            } else {
              lock(state.grid, state.active);
              state.pieces += 1;
              if (state.pieces % 20 === 0) {
                state.level += 1;
                state.gravity = gravityForLevel(state.level);
                updateLevel();
              }
              const cleared = clearRows(state.grid);
              if (cleared) {
                state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
                updateScore();
                recordClear(cleared);
              }
              if (state.grid[0].some((value) => value !== 0)) {
                triggerGameOver();
              } else {
                spawn();
                if (!canMove(state.grid, state.active, 0, 0)) {
                  triggerGameOver();
                }
              }
            }
          }
        }
      }
    } catch (err) {
      log(`Tick error: ${err && err.message ? err.message : err}`);
      try {
        triggerGameOver();
      } catch (_) {
        // ignore
      }
    } finally {
      try {
        if (state.renderEvery <= 1) {
          draw(state.grid, state.active);
        } else if (state.renderCounter <= 0) {
          draw(state.grid, state.active);
          state.renderCounter = state.renderEvery - 1;
        } else {
          state.renderCounter -= 1;
        }
      } catch (_) {
        // ignore rendering errors
      }
      scheduleNext();
    }
  }

  function start() {
    if (state.running) {
      log('Already running');
      return;
    }
    if (canvas && typeof canvas.focus === 'function') {
      try {
        // Prevent the focus call from forcing the page to scroll back to the canvas.
        canvas.focus({ preventScroll: true });
      } catch (err) {
        canvas.focus();
      }
    }
    state.grid = emptyGrid();
    state.score = 0;
    state.level = 0;
    state.pieces = 0;
    state.gravity = gravityForLevel(0);
    updateScore();
    updateLevel();
    state.last = 0;
    state.acc = 0;
    state.wdAcc = 0;
    state.lastSig = '';
    state.running = true;
    state.paused = false;
    updateRenderDecimation();
    spawn();
    draw(state.grid, state.active);
    scheduleNext();
    log('Game started');
    renderControls();
  }

  function pause() {
    if (!state.running) {
      log('Pause ignored: not running');
      return;
    }
    state.paused = true;
    log('Paused');
    renderControls();
  }

  function resume() {
    if (!state.running) {
      log('Resume ignored: not running');
      return;
    }
    state.paused = false;
    log('Resumed');
    renderControls();
  }

  function stop() {
    if (!state.running) {
      log('Stop ignored: not running');
      return;
    }
    state.running = false;
    state.paused = false;
    if (state.raf) {
      clearTimeout(state.raf);
      state.raf = null;
    }
    log('Game stopped');
    renderControls();
  }

  function renderControls() {
    if (!toggleBtn) return;
    const showPauseIcon = state.running && !state.paused;
    toggleBtn.textContent = showPauseIcon ? '⏸' : '▶';
    toggleBtn.classList.toggle('icon-btn--emerald', showPauseIcon);
  }

  function togglePlayPause() {
    if (!state.running) {
      start();
    } else if (state.paused) {
      resume();
    } else {
      pause();
    }
  }

  function resetGame() {
    if (state.running) {
      stop();
    }
    start();
  }

  function handleKeyDown(event) {
    if (!state.running) return;
    if (BLOCK_KEYS.has(event.key)) {
      event.preventDefault();
    }
    if (state.paused || !state.active || trainingHooks.isEnabled()) {
      return;
    }
    if (event.key === 'ArrowLeft' && canMove(state.grid, state.active, -1, 0)) {
      state.active.move(-1, 0);
    } else if (event.key === 'ArrowRight' && canMove(state.grid, state.active, 1, 0)) {
      state.active.move(1, 0);
    } else if (event.key === 'ArrowUp') {
      state.active.rotate();
      if (!canMove(state.grid, state.active, 0, 0)) {
        state.active.rotate(-1);
      }
    } else if (event.key === 'ArrowDown' && canMove(state.grid, state.active, 0, 1)) {
      state.active.move(0, 1);
    } else if (event.key === ' ') {
      while (canMove(state.grid, state.active, 0, 1)) {
        state.active.move(0, 1);
      }
      lock(state.grid, state.active);
      state.pieces += 1;
      if (state.pieces % 20 === 0) {
        state.level += 1;
        state.gravity = gravityForLevel(state.level);
        updateLevel();
      }
      const cleared = clearRows(state.grid);
      if (cleared) {
        state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
        updateScore();
        recordClear(cleared);
      }
      if (state.grid[0].some((value) => value !== 0)) {
        triggerGameOver();
      } else {
        spawn();
        if (!canMove(state.grid, state.active, 0, 0)) {
          triggerGameOver();
        }
      }
    }
    draw(state.grid, state.active);
  }

  function registerTrainingInterface(hooks) {
    trainingHooks = {
      aiStep: hooks && typeof hooks.aiStep === 'function' ? hooks.aiStep : () => {},
      isEnabled: hooks && typeof hooks.isEnabled === 'function' ? hooks.isEnabled : () => false,
      shouldSkipRendering: hooks && typeof hooks.shouldSkipRendering === 'function' ? hooks.shouldSkipRendering : () => false,
    };
  }

  function setGameOverHandler(handler) {
    gameOverHandler = typeof handler === 'function' ? handler : null;
  }

  if (speedSlider) {
    speedSlider.addEventListener('input', () => {
      speedMult = Number(speedSlider.value) || 1;
      if (speedDisplay) {
        speedDisplay.textContent = `${speedMult}x`;
      }
      if (state.running) {
        if (state.raf) {
          clearTimeout(state.raf);
          state.raf = null;
        }
        scheduleNext();
      }
    });
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      togglePlayPause();
      renderControls();
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      resetGame();
    });
  }

  document.addEventListener('keydown', handleKeyDown);

  renderer.setStateProvider(() => ({
    grid: state.grid,
    active: state.active,
    next: state.next,
  }));
  renderer.setSkipCallback(() => trainingHooks.shouldSkipRendering());
  renderer.applyCanvasScale();
  draw(state.grid, state.active);
  drawNext(state.next);
  requestAnimationFrame(() => {
    if (renderer.applyCanvasScale()) {
      draw(state.grid, state.active);
      drawNext(state.next);
    }
  });
  renderer.registerResizeListeners();

  return {
    state,
    start,
    stop,
    pause,
    resume,
    togglePlayPause,
    resetGame,
    spawn,
    triggerGameOver,
    updateScore,
    updateLevel,
    recordClear,
    canMove,
    lock,
    clearRows,
    gravityForLevel,
    registerTrainingInterface,
    setGameOverHandler,
    log,
  };
}
