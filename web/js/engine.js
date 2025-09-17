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
    this.shape = shape;
    this.rot = 0;
    this.row = 0;
    this.col = Math.floor(WIDTH / 2) - 2;
  }

  blocks() {
    const state = SHAPES[this.shape][this.rot];
    return state.map(([dr, dc]) => [this.row + dr, this.col + dc]);
  }

  move(dx, dy) {
    this.col += dx;
    this.row += dy;
  }

  rotate(dir = 1) {
    const states = SHAPES[this.shape];
    const next = dir >= 0 ? 1 : states.length - 1;
    this.rot = (this.rot + next) % states.length;
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
  const remaining = grid.filter((row) => row.some((value) => value === 0));
  const cleared = HEIGHT - remaining.length;
  while (remaining.length < HEIGHT) {
    remaining.unshift(Array(WIDTH).fill(0));
  }
  for (let r = 0; r < HEIGHT; r += 1) {
    grid[r] = remaining[r];
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
