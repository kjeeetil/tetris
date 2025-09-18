import { HEIGHT, WIDTH } from '../constants.js';
import { SHAPES } from '../engine.js';

const DEFAULT_CONV_FILTERS = [32, 64];
const DEFAULT_KERNEL_SIZE = 3;
const DEFAULT_DENSE_UNITS = 128;
const DEFAULT_POLICY_DENSE_UNITS = 64;
const DEFAULT_VALUE_DENSE_UNITS = 64;
const DEFAULT_ACTIVATION = 'relu';
const DEFAULT_VALUE_ACTIVATION = 'tanh';
const DEFAULT_POOL_SIZE = [2, 2];

const MODEL_CACHE = new Map();
const PREDICT_FUNCTION_CACHE = new WeakMap();

export const ALPHA_BOARD_WIDTH = WIDTH;
export const ALPHA_BOARD_HEIGHT = HEIGHT;
export const ALPHA_BOARD_CHANNELS = 3;
export const ALPHA_PREVIEW_SLOTS = 2;
export const ALPHA_PIECES = Object.freeze(['I', 'O', 'T', 'S', 'Z', 'J', 'L']);
const PIECE_INDEX = new Map(ALPHA_PIECES.map((shape, index) => [shape, index]));

export const ALPHA_ACTIONS = (() => {
  const actions = [];
  for (let rotation = 0; rotation < 4; rotation += 1) {
    for (let column = 0; column < ALPHA_BOARD_WIDTH; column += 1) {
      actions.push({ rotation, column });
    }
  }
  return actions;
})();

export const ALPHA_POLICY_SIZE = ALPHA_ACTIONS.length;
const ALPHA_HEURISTIC_FEATURES = 6;
const ALPHA_GAME_SCALARS = 4;
const PIECE_FEATURES = ALPHA_PIECES.length * (1 + ALPHA_PREVIEW_SLOTS);
export const ALPHA_AUX_FEATURE_COUNT = PIECE_FEATURES + ALPHA_HEURISTIC_FEATURES + ALPHA_GAME_SCALARS;

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function resolveTensorFlow(explicit) {
  if (explicit) {
    return explicit;
  }
  if (typeof globalThis !== 'undefined' && globalThis.tf) {
    return globalThis.tf;
  }
  if (typeof window !== 'undefined' && window.tf) {
    return window.tf;
  }
  throw new Error('TensorFlow.js is not loaded. Ensure loader.js has completed before creating models.');
}

function normalizeModelConfig(config = {}) {
  const convFilters = Array.isArray(config.convFilters) && config.convFilters.length
    ? config.convFilters.map((value) => Math.max(1, Math.floor(value)))
    : DEFAULT_CONV_FILTERS.slice();
  const baseKernelSize = Number.isFinite(config.kernelSize) ? Math.max(1, Math.floor(config.kernelSize)) : DEFAULT_KERNEL_SIZE;
  const convKernelSizes = Array.isArray(config.convKernelSizes) ? config.convKernelSizes : [];
  const convStrides = Array.isArray(config.convStrides) ? config.convStrides : [];
  const convPadding = typeof config.convPadding === 'string' ? config.convPadding : 'same';
  const convDropoutRate = Number.isFinite(config.convDropoutRate) ? clamp01(config.convDropoutRate) : 0;
  const poolEvery = Number.isFinite(config.poolEvery) ? Math.max(0, Math.floor(config.poolEvery)) : 0;
  const poolSize = Array.isArray(config.poolSize) && config.poolSize.length >= 2
    ? [Math.max(1, Math.floor(config.poolSize[0])), Math.max(1, Math.floor(config.poolSize[1]))]
    : Number.isFinite(config.poolSize)
      ? [Math.max(1, Math.floor(config.poolSize)), Math.max(1, Math.floor(config.poolSize))]
      : DEFAULT_POOL_SIZE.slice();
  const poolStride = Array.isArray(config.poolStride) && config.poolStride.length >= 2
    ? [Math.max(1, Math.floor(config.poolStride[0])), Math.max(1, Math.floor(config.poolStride[1]))]
    : Number.isFinite(config.poolStride)
      ? [Math.max(1, Math.floor(config.poolStride)), Math.max(1, Math.floor(config.poolStride))]
      : poolSize.slice();

  const convLayers = convFilters.map((filters, index) => ({
    filters,
    kernelSize: Math.max(1, Math.floor(convKernelSizes[index] || baseKernelSize)),
    strides: Math.max(1, Math.floor(convStrides[index] || 1)),
    padding: convPadding,
  }));

  return {
    convLayers,
    convDropoutRate,
    activation: typeof config.activation === 'string' ? config.activation : DEFAULT_ACTIVATION,
    denseUnits: Number.isFinite(config.denseUnits) && config.denseUnits > 0
      ? Math.floor(config.denseUnits)
      : DEFAULT_DENSE_UNITS,
    auxUnits: Number.isFinite(config.auxUnits) && config.auxUnits > 0
      ? Math.floor(config.auxUnits)
      : DEFAULT_DENSE_UNITS,
    policyDenseUnits: Number.isFinite(config.policyDenseUnits) && config.policyDenseUnits >= 0
      ? Math.floor(config.policyDenseUnits)
      : DEFAULT_POLICY_DENSE_UNITS,
    valueDenseUnits: Number.isFinite(config.valueDenseUnits) && config.valueDenseUnits >= 0
      ? Math.floor(config.valueDenseUnits)
      : DEFAULT_VALUE_DENSE_UNITS,
    dropoutRate: Number.isFinite(config.dropoutRate) ? clamp01(config.dropoutRate) : 0,
    useBatchNorm: !!config.useBatchNorm,
    poolEvery,
    poolSize,
    poolStride,
    valueActivation: typeof config.valueActivation === 'string' ? config.valueActivation : DEFAULT_VALUE_ACTIVATION,
    boardChannels: ALPHA_BOARD_CHANNELS,
  };
}

function configSignature(normalizedConfig) {
  const serializable = {
    convLayers: normalizedConfig.convLayers.map((layer) => ({
      filters: layer.filters,
      kernelSize: layer.kernelSize,
      strides: layer.strides,
      padding: layer.padding,
    })),
    convDropoutRate: normalizedConfig.convDropoutRate,
    activation: normalizedConfig.activation,
    denseUnits: normalizedConfig.denseUnits,
    auxUnits: normalizedConfig.auxUnits,
    policyDenseUnits: normalizedConfig.policyDenseUnits,
    valueDenseUnits: normalizedConfig.valueDenseUnits,
    dropoutRate: normalizedConfig.dropoutRate,
    useBatchNorm: normalizedConfig.useBatchNorm,
    poolEvery: normalizedConfig.poolEvery,
    poolSize: normalizedConfig.poolSize,
    poolStride: normalizedConfig.poolStride,
    valueActivation: normalizedConfig.valueActivation,
    boardChannels: normalizedConfig.boardChannels,
  };
  return JSON.stringify(serializable);
}

function getCachedModel(signature) {
  const entry = MODEL_CACHE.get(signature);
  if (!entry || !entry.model || entry.disposed) {
    return null;
  }
  return entry.model;
}

function cacheModel(signature, model) {
  const entry = { model, disposed: false };
  MODEL_CACHE.set(signature, entry);
  const originalDispose = model.dispose.bind(model);
  model.dispose = function patchedDispose(...args) {
    entry.disposed = true;
    MODEL_CACHE.delete(signature);
    PREDICT_FUNCTION_CACHE.delete(model);
    return originalDispose(...args);
  };
}

function getPredictFunction(model) {
  if (PREDICT_FUNCTION_CACHE.has(model)) {
    return PREDICT_FUNCTION_CACHE.get(model);
  }
  if (!model || typeof model.predict !== 'function') {
    throw new Error('Provided model does not implement predict().');
  }
  const fn = model.predict.bind(model);
  PREDICT_FUNCTION_CACHE.set(model, fn);
  return fn;
}

function computeColumnStats(grid) {
  const heights = new Array(ALPHA_BOARD_WIDTH).fill(0);
  const columnHoles = new Array(ALPHA_BOARD_WIDTH).fill(0);
  let aggregateHeight = 0;
  let maxHeight = 0;
  let filledCells = 0;
  for (let col = 0; col < ALPHA_BOARD_WIDTH; col += 1) {
    let columnHeight = 0;
    let seenBlock = false;
    let holes = 0;
    for (let row = 0; row < ALPHA_BOARD_HEIGHT; row += 1) {
      const cell = grid[row] && grid[row][col] ? 1 : 0;
      if (cell) {
        filledCells += 1;
        if (!seenBlock) {
          columnHeight = ALPHA_BOARD_HEIGHT - row;
          seenBlock = true;
        }
      } else if (seenBlock) {
        holes += 1;
      }
    }
    heights[col] = columnHeight;
    columnHoles[col] = holes;
    aggregateHeight += columnHeight;
    if (columnHeight > maxHeight) {
      maxHeight = columnHeight;
    }
  }
  let bumpiness = 0;
  for (let col = 0; col < ALPHA_BOARD_WIDTH - 1; col += 1) {
    bumpiness += Math.abs(heights[col] - heights[col + 1]);
  }
  let wells = 0;
  for (let col = 0; col < ALPHA_BOARD_WIDTH; col += 1) {
    const left = col > 0 ? heights[col - 1] : heights[col];
    const right = col < ALPHA_BOARD_WIDTH - 1 ? heights[col + 1] : heights[col];
    const minAdjacent = Math.min(left, right);
    if (minAdjacent > heights[col]) {
      wells += minAdjacent - heights[col];
    }
  }
  const totalCells = ALPHA_BOARD_WIDTH * ALPHA_BOARD_HEIGHT;
  return {
    heights,
    columnHoles,
    aggregateHeight,
    maxHeight,
    filledCells,
    bumpiness,
    wells,
    holes: columnHoles.reduce((acc, value) => acc + value, 0),
    totalCells,
  };
}

function fillBoardTensor(boardBuffer, grid, active) {
  const stride = ALPHA_BOARD_CHANNELS;
  const stats = computeColumnStats(grid);
  for (let row = 0; row < ALPHA_BOARD_HEIGHT; row += 1) {
    for (let col = 0; col < ALPHA_BOARD_WIDTH; col += 1) {
      const cell = grid[row] && grid[row][col] ? 1 : 0;
      const baseIndex = (row * ALPHA_BOARD_WIDTH + col) * stride;
      boardBuffer[baseIndex] = cell;
      const heightNorm = stats.heights[col] / ALPHA_BOARD_HEIGHT;
      boardBuffer[baseIndex + 2] = heightNorm;
    }
  }

  if (active && typeof active.blocks === 'function') {
    const blocks = active.blocks();
    for (let i = 0; i < blocks.length; i += 1) {
      const [row, col] = blocks[i];
      if (row < 0 || row >= ALPHA_BOARD_HEIGHT || col < 0 || col >= ALPHA_BOARD_WIDTH) {
        continue;
      }
      const index = (row * ALPHA_BOARD_WIDTH + col) * stride;
      boardBuffer[index + 1] = 1;
    }
  }
  return stats;
}

function encodePieceOneHot(shape, target, offset = 0) {
  if (!target || target.length < offset + ALPHA_PIECES.length) {
    return;
  }
  if (typeof shape !== 'string') {
    return;
  }
  const index = PIECE_INDEX.get(shape);
  if (index === undefined) {
    return;
  }
  target[offset + index] = 1;
}

function normalizeScore(score) {
  if (!Number.isFinite(score) || score <= 0) {
    return 0;
  }
  const scaled = score / (score + 5000);
  return clamp01(scaled);
}

function normalizePieces(pieces) {
  if (!Number.isFinite(pieces) || pieces <= 0) {
    return 0;
  }
  const scaled = pieces / (pieces + 200);
  return clamp01(scaled);
}

function normalizeGravity(gravity) {
  if (!Number.isFinite(gravity) || gravity <= 0) {
    return 0;
  }
  const clamped = Math.min(Math.max(gravity, 1), 120);
  return clamp01(1 - clamped / 120);
}

function computeActionMask(shape) {
  const mask = new Float32Array(ALPHA_POLICY_SIZE);
  if (!shape || !SHAPES[shape]) {
    return mask;
  }
  const states = SHAPES[shape];
  for (let actionIndex = 0; actionIndex < ALPHA_ACTIONS.length; actionIndex += 1) {
    const action = ALPHA_ACTIONS[actionIndex];
    const rotation = action.rotation % states.length;
    const state = states[rotation];
    let maxC = 0;
    for (let i = 0; i < state.length; i += 1) {
      const [, col] = state[i];
      if (col > maxC) {
        maxC = col;
      }
    }
    const width = maxC + 1;
    if (action.column <= ALPHA_BOARD_WIDTH - width) {
      mask[actionIndex] = 1;
    }
  }
  return mask;
}

export function prepareAlphaInputs(gameState = {}, options = {}) {
  const board = new Float32Array(ALPHA_BOARD_HEIGHT * ALPHA_BOARD_WIDTH * ALPHA_BOARD_CHANNELS);
  const grid = Array.isArray(gameState.grid) ? gameState.grid : [];
  const active = gameState.active || null;
  const stats = fillBoardTensor(board, grid, active);

  const aux = new Float32Array(ALPHA_AUX_FEATURE_COUNT);
  let offset = 0;

  encodePieceOneHot(active && active.shape, aux, offset);
  offset += ALPHA_PIECES.length;

  const previews = [];
  if (typeof gameState.next === 'string') {
    previews.push(gameState.next);
  }
  if (Array.isArray(gameState.nextQueue)) {
    for (let i = 0; i < gameState.nextQueue.length; i += 1) {
      previews.push(gameState.nextQueue[i]);
    }
  }
  if (Array.isArray(gameState.preview)) {
    for (let i = 0; i < gameState.preview.length; i += 1) {
      previews.push(gameState.preview[i]);
    }
  }
  for (let slot = 0; slot < ALPHA_PREVIEW_SLOTS; slot += 1) {
    const shape = previews[slot];
    encodePieceOneHot(shape, aux, offset);
    offset += ALPHA_PIECES.length;
  }

  const aggregateHeightNorm = stats.totalCells > 0 ? stats.aggregateHeight / stats.totalCells : 0;
  const maxHeightNorm = stats.maxHeight / ALPHA_BOARD_HEIGHT;
  const holesNorm = stats.totalCells > 0 ? stats.holes / stats.totalCells : 0;
  const bumpinessNorm = (ALPHA_BOARD_WIDTH > 1)
    ? stats.bumpiness / ((ALPHA_BOARD_WIDTH - 1) * ALPHA_BOARD_HEIGHT)
    : 0;
  const wellsNorm = stats.totalCells > 0 ? stats.wells / stats.totalCells : 0;
  const filledNorm = stats.totalCells > 0 ? stats.filledCells / stats.totalCells : 0;

  aux[offset + 0] = clamp01(aggregateHeightNorm);
  aux[offset + 1] = clamp01(maxHeightNorm);
  aux[offset + 2] = clamp01(holesNorm);
  aux[offset + 3] = clamp01(bumpinessNorm);
  aux[offset + 4] = clamp01(wellsNorm);
  aux[offset + 5] = clamp01(filledNorm);
  offset += ALPHA_HEURISTIC_FEATURES;

  const level = Number.isFinite(gameState.level) ? gameState.level : 0;
  const score = Number.isFinite(gameState.score) ? gameState.score : 0;
  const pieces = Number.isFinite(gameState.pieces) ? gameState.pieces : 0;
  const gravity = Number.isFinite(gameState.gravity) ? gameState.gravity : 0;

  aux[offset + 0] = clamp01(level / 20);
  aux[offset + 1] = normalizeScore(score);
  aux[offset + 2] = normalizePieces(pieces);
  aux[offset + 3] = normalizeGravity(gravity);

  const policyMask = computeActionMask(active && active.shape);

  let boardTensor = null;
  let auxTensor = null;
  if (options.asTensors) {
    const tf = resolveTensorFlow(options.tf);
    boardTensor = tf.tensor(board, [1, ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, ALPHA_BOARD_CHANNELS], 'float32');
    auxTensor = tf.tensor(aux, [1, ALPHA_AUX_FEATURE_COUNT], 'float32');
  }

  return {
    board,
    aux,
    boardTensor,
    auxTensor,
    boardShape: [1, ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, ALPHA_BOARD_CHANNELS],
    auxShape: [1, ALPHA_AUX_FEATURE_COUNT],
    policyMask,
    activeShape: active ? active.shape : null,
    preview: previews.slice(0, ALPHA_PREVIEW_SLOTS),
  };
}

export function buildAlphaTetrisModel(config = {}) {
  const tf = resolveTensorFlow(config.tf);
  const normalized = normalizeModelConfig(config);
  const signature = configSignature(normalized);
  const cachedModel = getCachedModel(signature);
  if (cachedModel) {
    return cachedModel;
  }

  const boardInput = tf.input({
    name: 'board',
    shape: [ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, normalized.boardChannels],
  });
  let boardPath = boardInput;

  normalized.convLayers.forEach((layerConfig, index) => {
    const conv = tf.layers.conv2d({
      filters: layerConfig.filters,
      kernelSize: layerConfig.kernelSize,
      strides: layerConfig.strides,
      padding: layerConfig.padding,
      useBias: !normalized.useBatchNorm,
      kernelInitializer: 'heNormal',
    });
    boardPath = conv.apply(boardPath);
    if (normalized.useBatchNorm) {
      boardPath = tf.layers.batchNormalization().apply(boardPath);
    }
    boardPath = tf.layers.activation({ activation: normalized.activation }).apply(boardPath);
    if (normalized.convDropoutRate > 0) {
      boardPath = tf.layers.dropout({ rate: normalized.convDropoutRate }).apply(boardPath);
    }
    if (normalized.poolEvery && (index + 1) % normalized.poolEvery === 0) {
      boardPath = tf.layers.maxPooling2d({
        poolSize: normalized.poolSize,
        strides: normalized.poolStride,
        padding: 'valid',
      }).apply(boardPath);
    }
  });

  boardPath = tf.layers.flatten().apply(boardPath);

  const auxInput = tf.input({ name: 'aux', shape: [ALPHA_AUX_FEATURE_COUNT] });
  let auxPath = auxInput;
  if (normalized.auxUnits > 0) {
    auxPath = tf.layers.dense({
      units: normalized.auxUnits,
      activation: normalized.activation,
      kernelInitializer: 'heNormal',
    }).apply(auxPath);
  }

  let merged = tf.layers.concatenate().apply([boardPath, auxPath]);
  if (normalized.denseUnits > 0) {
    merged = tf.layers.dense({
      units: normalized.denseUnits,
      activation: normalized.activation,
      kernelInitializer: 'heNormal',
    }).apply(merged);
    if (normalized.dropoutRate > 0) {
      merged = tf.layers.dropout({ rate: normalized.dropoutRate }).apply(merged);
    }
  }

  let policyBranch = merged;
  if (normalized.policyDenseUnits > 0) {
    policyBranch = tf.layers.dense({
      units: normalized.policyDenseUnits,
      activation: normalized.activation,
      kernelInitializer: 'heNormal',
    }).apply(policyBranch);
  }
  const policyOutput = tf.layers.dense({
    units: ALPHA_POLICY_SIZE,
    activation: 'linear',
    name: 'policy',
    kernelInitializer: 'glorotUniform',
  }).apply(policyBranch);

  let valueBranch = merged;
  if (normalized.valueDenseUnits > 0) {
    valueBranch = tf.layers.dense({
      units: normalized.valueDenseUnits,
      activation: normalized.activation,
      kernelInitializer: 'heNormal',
    }).apply(valueBranch);
  }
  const valueOutput = tf.layers.dense({
    units: 1,
    activation: normalized.valueActivation,
    name: 'value',
    kernelInitializer: 'glorotUniform',
  }).apply(valueBranch);

  const model = tf.model({
    name: 'AlphaTetrisModel',
    inputs: [boardInput, auxInput],
    outputs: [policyOutput, valueOutput],
  });

  cacheModel(signature, model);
  return model;
}

export function runAlphaInference(model, inputs = {}, options = {}) {
  if (!model) {
    throw new Error('A TensorFlow.js model instance is required for inference.');
  }
  const hasBoardTensor = inputs && inputs.boardTensor;
  const hasAuxTensor = inputs && inputs.auxTensor;
  if (!hasBoardTensor && (!inputs || !inputs.board)) {
    throw new Error('AlphaTetris inference requires prepared board inputs.');
  }
  if (!hasAuxTensor && (!inputs || !inputs.aux)) {
    throw new Error('AlphaTetris inference requires prepared auxiliary inputs.');
  }

  const tf = resolveTensorFlow(options.tf);
  const reuseGraph = options.reuseGraph !== false;

  const boardShape = inputs.boardShape
    || (hasBoardTensor ? inputs.boardTensor.shape : [1, ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, ALPHA_BOARD_CHANNELS]);
  const auxShape = inputs.auxShape
    || (hasAuxTensor ? inputs.auxTensor.shape : [1, ALPHA_AUX_FEATURE_COUNT]);

  const boardTensor = hasBoardTensor
    ? inputs.boardTensor
    : tf.tensor(inputs.board, boardShape, 'float32');
  const auxTensor = hasAuxTensor
    ? inputs.auxTensor
    : tf.tensor(inputs.aux, auxShape, 'float32');

  const predictFn = reuseGraph ? getPredictFunction(model) : model.predict.bind(model);

  let rawOutput;
  let policyTensor;
  let valueTensor;
  try {
    rawOutput = predictFn([boardTensor, auxTensor]);
    const outputs = Array.isArray(rawOutput) ? rawOutput : [rawOutput];
    policyTensor = outputs[0];
    valueTensor = outputs[1] || null;

    const batchSize = policyTensor && policyTensor.shape && policyTensor.shape.length > 0
      ? policyTensor.shape[0]
      : 1;
    const actionCount = policyTensor && policyTensor.shape && policyTensor.shape.length >= 2
      ? policyTensor.shape[1]
      : ALPHA_POLICY_SIZE;

    const logitsData = policyTensor.dataSync();
    const policyLogits = new Array(batchSize);
    for (let i = 0; i < batchSize; i += 1) {
      const start = i * actionCount;
      const end = start + actionCount;
      const slice = typeof logitsData.subarray === 'function'
        ? logitsData.subarray(start, end)
        : Array.prototype.slice.call(logitsData, start, end);
      const sample = new Float32Array(actionCount);
      sample.set(slice);
      policyLogits[i] = sample;
    }

    let values = null;
    if (valueTensor) {
      const valueData = valueTensor.dataSync();
      const total = valueData.length;
      const stride = Math.max(1, Math.floor(total / Math.max(1, batchSize)));
      values = new Float32Array(batchSize);
      for (let i = 0; i < batchSize; i += 1) {
        const idx = Math.min(total - 1, i * stride);
        const raw = valueData[idx];
        values[i] = Number.isFinite(raw) ? raw : 0;
      }
    }

    return {
      batchSize,
      policyLogits,
      values,
      value: values && values.length ? values[0] : 0,
    };
  } finally {
    if (!hasBoardTensor && boardTensor && typeof boardTensor.dispose === 'function') {
      boardTensor.dispose();
    }
    if (!hasAuxTensor && auxTensor && typeof auxTensor.dispose === 'function') {
      auxTensor.dispose();
    }
    if (policyTensor && typeof policyTensor.dispose === 'function') {
      policyTensor.dispose();
    }
    if (valueTensor && valueTensor !== policyTensor && typeof valueTensor.dispose === 'function') {
      valueTensor.dispose();
    }
    if (rawOutput) {
      if (Array.isArray(rawOutput)) {
        for (let i = 0; i < rawOutput.length; i += 1) {
          const tensor = rawOutput[i];
          if (!tensor || tensor === policyTensor || tensor === valueTensor) {
            continue;
          }
          if (typeof tensor.dispose === 'function') {
            tensor.dispose();
          }
        }
      } else if (rawOutput !== policyTensor && rawOutput !== valueTensor && typeof rawOutput.dispose === 'function') {
        rawOutput.dispose();
      }
    }
  }
}

