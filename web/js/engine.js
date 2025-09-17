import { HEIGHT, LEVEL_FRAMES, WIDTH } from './constants.js';

const BASE_SHAPES = {
  I: [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
  ],
  O: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ],
  T: [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 1],
  ],
  S: [
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
  ],
  Z: [
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 2],
  ],
  J: [
    [0, 0],
    [1, 0],
    [1, 1],
    [1, 2],
  ],
  L: [
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
  ],
};

function rotate(state) {
  const rotated = state.map(([r, c]) => [c, -r]);
  const minR = Math.min(...rotated.map(([r]) => r));
  const minC = Math.min(...rotated.map(([, c]) => c));
  return rotated.map(([r, c]) => [r - minR, c - minC]);
}

function genRotations(shape) {
  const states = [shape];
  let current = shape;
  for (let i = 0; i < 3; i += 1) {
    current = rotate(current);
    states.push(current);
  }
  return states;
}

export const SHAPES = Object.fromEntries(
  Object.entries(BASE_SHAPES).map(([key, state]) => [key, genRotations(state)]),
);

export const UNIQUE_ROTATIONS = Object.fromEntries(
  Object.entries(SHAPES).map(([shape, states]) => {
    const seen = new Set();
    const indices = [];
    for (let i = 0; i < states.length; i += 1) {
      const signature = states[i]
        .map(([r, c]) => `${r},${c}`)
        .sort()
        .join('|');
      if (!seen.has(signature)) {
        seen.add(signature);
        indices.push(i);
      }
    }
    return [shape, indices];
  }),
);

export class Piece {
  constructor(shape) {
    this._row = 0;
    this._col = Math.floor(WIDTH / 2) - 2;
    this._rot = 0;
    this._shape = 'I';
    this._dirty = true;
    this._blocks = Array.from({ length: 4 }, () => [0, 0]);
    this.shape = shape && SHAPES[shape] ? shape : 'I';
    this.rot = 0;
    this.row = 0;
    this.col = Math.floor(WIDTH / 2) - 2;
  }

  get shape() {
    return this._shape;
  }

  set shape(value) {
    if (!value || !SHAPES[value]) {
      return;
    }
    if (value !== this._shape) {
      this._shape = value;
      const states = SHAPES[value];
      if (states && states.length) {
        const len = states.length;
        if (this._rot >= len || this._rot < 0) {
          this._rot = ((this._rot % len) + len) % len;
        }
      } else {
        this._rot = 0;
      }
      this._dirty = true;
    }
  }

  get rot() {
    return this._rot;
  }

  set rot(value) {
    const states = SHAPES[this._shape];
    const len = states && states.length ? states.length : 1;
    const raw = Number.isFinite(value) ? value : 0;
    let next = raw % len;
    if (next < 0) {
      next += len;
    }
    if (next !== this._rot) {
      this._rot = next;
      this._dirty = true;
    }
  }

  get row() {
    return this._row;
  }

  set row(value) {
    const next = Number.isFinite(value) ? Math.round(value) : this._row;
    if (next !== this._row) {
      this._row = next;
      this._dirty = true;
    }
  }

  get col() {
    return this._col;
  }

  set col(value) {
    const next = Number.isFinite(value) ? Math.round(value) : this._col;
    if (next !== this._col) {
      this._col = next;
      this._dirty = true;
    }
  }

  blocks() {
    if (this._dirty) {
      this._recomputeBlocks();
    }
    return this._blocks;
  }

  move(dx, dy) {
    const nextCol = Number.isFinite(dx) ? this._col + dx : this._col;
    const nextRow = Number.isFinite(dy) ? this._row + dy : this._row;
    this.col = nextCol;
    this.row = nextRow;
  }

  rotate(dir = 1) {
    const states = SHAPES[this._shape];
    if (!states || !states.length) {
      return;
    }
    const step = dir >= 0 ? 1 : -1;
    this.rot = this._rot + step;
  }

  _recomputeBlocks() {
    const states = SHAPES[this._shape];
    const state = states && states.length ? states[this._rot] : null;
    if (!state) {
      this._dirty = false;
      return this._blocks;
    }
    const rowBase = this._row;
    const colBase = this._col;
    if (!this._blocks || this._blocks.length < state.length) {
      this._blocks = Array.from({ length: state.length }, () => [0, 0]);
    }
    for (let i = 0; i < state.length; i += 1) {
      const block = this._blocks[i] || (this._blocks[i] = [0, 0]);
      const [dr, dc] = state[i];
      block[0] = rowBase + dr;
      block[1] = colBase + dc;
    }
    this._dirty = false;
    return this._blocks;
  }
}

export function emptyGrid() {
  return Array.from({ length: HEIGHT }, () => Array(WIDTH).fill(0));
}

export function canMove(grid, piece, dx, dy) {
  for (const [r, c] of piece.blocks()) {
    const nr = r + dy;
    const nc = c + dx;
    if (nr < 0 || nr >= HEIGHT || nc < 0 || nc >= WIDTH) {
      return false;
    }
    if (grid[nr][nc] !== 0) {
      return false;
    }
  }
  return true;
}

export function lock(grid, piece) {
  for (const [r, c] of piece.blocks()) {
    grid[r][c] = piece.shape;
  }
}

export function clearRows(grid) {
  let write = HEIGHT - 1;
  let cleared = 0;

  for (let row = HEIGHT - 1; row >= 0; row -= 1) {
    const source = grid[row];
    let filled = true;
    for (let col = 0; col < WIDTH; col += 1) {
      if (!source[col]) {
        filled = false;
        break;
      }
    }

    if (filled) {
      cleared += 1;
      continue;
    }

    if (write !== row) {
      const target = grid[write];
      for (let col = 0; col < WIDTH; col += 1) {
        target[col] = source[col];
      }
    }
    write -= 1;
  }

  for (; write >= 0; write -= 1) {
    const target = grid[write];
    for (let col = 0; col < WIDTH; col += 1) {
      target[col] = 0;
    }
  }

  return cleared;
}

export function gravityForLevel(level) {
  if (level >= 10) {
    return 0;
  }
  const frames = LEVEL_FRAMES[level] ?? LEVEL_FRAMES[9];
  return (frames / 60) * 1000;
}

export function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}
