import test from 'node:test';
import assert from 'node:assert/strict';

import {
  prepareAlphaInputs,
  ALPHA_BOARD_WIDTH,
  ALPHA_BOARD_HEIGHT,
} from '../alphatetris.js';

const ENGINEERED_FEATURE_LENGTH = 17;

function createSampleGrid() {
  const grid = Array.from({ length: ALPHA_BOARD_HEIGHT }, () => Array(ALPHA_BOARD_WIDTH).fill(0));
  const bottom = ALPHA_BOARD_HEIGHT - 1;
  grid[bottom][0] = 1;
  grid[bottom - 1][0] = 1;
  grid[bottom][1] = 1;
  grid[bottom - 2][2] = 1;
  grid[bottom - 3][2] = 1;
  grid[bottom - 4][2] = 1;
  grid[bottom][7] = 1;
  grid[bottom - 1][7] = 1;
  grid[bottom - 2][7] = 1;
  grid[bottom - 3][7] = 1;
  grid[bottom][9] = 1;
  return grid;
}

function computeColumnMetrics(grid) {
  const masks = typeof Uint32Array !== 'undefined'
    ? new Uint32Array(ALPHA_BOARD_WIDTH)
    : new Array(ALPHA_BOARD_WIDTH).fill(0);
  const heights = new Array(ALPHA_BOARD_WIDTH).fill(0);
  for (let row = 0; row < ALPHA_BOARD_HEIGHT; row += 1) {
    const rowData = grid[row];
    for (let col = 0; col < ALPHA_BOARD_WIDTH; col += 1) {
      if (rowData[col]) {
        masks[col] |= 1 << (ALPHA_BOARD_HEIGHT - 1 - row);
      }
    }
  }
  for (let col = 0; col < ALPHA_BOARD_WIDTH; col += 1) {
    const mask = masks[col];
    if (mask) {
      heights[col] = (31 - Math.clz32(mask)) + 1;
    }
  }
  return { masks, heights };
}

function engineeredFeatureVector() {
  const feats = new Float32Array(ENGINEERED_FEATURE_LENGTH);
  for (let i = 0; i < feats.length; i += 1) {
    feats[i] = (i + 1) / (feats.length + 1);
  }
  return feats;
}

test('prepareAlphaInputs reuses provided column heights and masks', () => {
  const grid = createSampleGrid();
  const engineeredFeatures = engineeredFeatureVector();
  const baseline = prepareAlphaInputs({ grid, engineeredFeatures });
  const { masks, heights } = computeColumnMetrics(grid);
  const heightTyped = typeof Uint8Array !== 'undefined' ? Uint8Array.from(heights) : heights.slice();
  const maskTyped = typeof Uint32Array !== 'undefined' ? Uint32Array.from(masks) : masks.slice();
  const fast = prepareAlphaInputs({ grid, engineeredFeatures }, {
    columnHeights: heightTyped,
    columnMasks: maskTyped,
  });

  assert.equal(fast.board.length, baseline.board.length);
  for (let i = 0; i < fast.board.length; i += 1) {
    assert.strictEqual(fast.board[i], baseline.board[i]);
  }
  assert.equal(fast.aux.length, baseline.aux.length);
  for (let i = 0; i < fast.aux.length; i += 1) {
    assert.strictEqual(fast.aux[i], baseline.aux[i]);
  }
});

test('prepareAlphaInputs derives heights from provided column masks when needed', () => {
  const grid = createSampleGrid();
  const engineeredFeatures = engineeredFeatureVector();
  const baseline = prepareAlphaInputs({ grid, engineeredFeatures });
  const { masks } = computeColumnMetrics(grid);
  const maskTyped = typeof Uint32Array !== 'undefined' ? Uint32Array.from(masks) : masks.slice();
  const fast = prepareAlphaInputs({ grid, engineeredFeatures }, {
    columnMasks: maskTyped,
  });

  assert.equal(fast.board.length, baseline.board.length);
  for (let i = 0; i < fast.board.length; i += 1) {
    assert.strictEqual(fast.board[i], baseline.board[i]);
  }
});
