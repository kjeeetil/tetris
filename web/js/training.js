import { HEIGHT, MAX_AI_STEPS_PER_FRAME, WIDTH } from './constants.js';
import { Piece, SHAPES, UNIQUE_ROTATIONS, canMove, clearRows, emptyGrid, gravityForLevel, lock } from './engine.js';
import {
  ALPHA_ACTIONS,
  ALPHA_AUX_FEATURE_COUNT,
  ALPHA_BOARD_CHANNELS,
  ALPHA_BOARD_HEIGHT,
  ALPHA_BOARD_WIDTH,
  ALPHA_POLICY_SIZE,
  buildAlphaTetrisModel,
  prepareAlphaInputs,
  runAlphaInference,
} from './models/alphatetris.js';
import {
  backpropagate,
  computeVisitPolicy,
  createNode,
  normalizePriors,
  sampleFromPolicy,
  selectChild,
  softmax,
  visitStats,
} from './mcts.js';

let randnSpare = null;

  function arrayBufferToBase64(buffer) {
    if (!buffer) {
      return '';
    }
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(bytes.length, i + chunkSize));
    binary += String.fromCharCode.apply(null, chunk);
  }
  if (typeof btoa === 'function') {
    return btoa(binary);
  }
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(binary, 'binary').toString('base64');
  }
  throw new Error('Base64 encoding is not supported in this environment.');
}

  function base64ToArrayBuffer(base64) {
    if (!base64 || typeof base64 !== 'string') {
      return new ArrayBuffer(0);
  }
  let binary;
  if (typeof atob === 'function') {
    binary = atob(base64);
  } else if (typeof Buffer !== 'undefined') {
    binary = Buffer.from(base64, 'base64').toString('binary');
  } else {
    throw new Error('Base64 decoding is not supported in this environment.');
  }
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function countWeightsFromSpecs(weightSpecs) {
    if (!Array.isArray(weightSpecs) || !weightSpecs.length) {
      return 0;
    }
    let total = 0;
    for (let i = 0; i < weightSpecs.length; i += 1) {
      const spec = weightSpecs[i];
      if (!spec || !spec.shape) {
        total += 1;
        continue;
      }
      const shape = Array.isArray(spec.shape) ? spec.shape : [];
      if (!shape.length) {
        total += 1;
        continue;
      }
      let size = 1;
      for (let j = 0; j < shape.length; j += 1) {
        const dimRaw = shape[j];
        const dim = Number.isFinite(dimRaw) ? Math.max(1, Math.floor(dimRaw)) : 1;
        size *= dim;
      }
      total += size;
    }
    return total;
  }

  function cloneAlphaConvSummary(summary) {
    if (!summary || typeof summary !== 'object') {
      return null;
    }
    const cloned = {
      step: Number.isFinite(summary.step) ? summary.step : null,
      layers: [],
    };
    if (!Array.isArray(summary.layers) || !summary.layers.length) {
      return cloned;
    }
    for (let i = 0; i < summary.layers.length; i += 1) {
      const layer = summary.layers[i];
      if (!layer || typeof layer !== 'object') {
        continue;
      }
      const filters = [];
      if (Array.isArray(layer.filters)) {
        for (let j = 0; j < layer.filters.length; j += 1) {
          const filter = layer.filters[j];
          if (!filter || typeof filter !== 'object') {
            continue;
          }
          const values = filter.values instanceof Float32Array
            ? new Float32Array(filter.values)
            : Array.isArray(filter.values)
              ? Float32Array.from(filter.values)
              : null;
          filters.push({
            index: Number.isFinite(filter.index) ? filter.index : j,
            min: Number.isFinite(filter.min) ? filter.min : null,
            max: Number.isFinite(filter.max) ? filter.max : null,
            maxAbs: Number.isFinite(filter.maxAbs) ? filter.maxAbs : null,
            values,
          });
        }
      }
      cloned.layers.push({
        name: typeof layer.name === 'string' ? layer.name : `Conv ${i + 1}`,
        kernelSize: Array.isArray(layer.kernelSize) ? layer.kernelSize.slice() : null,
        inChannels: Number.isFinite(layer.inChannels) ? layer.inChannels : null,
        outChannels: Number.isFinite(layer.outChannels) ? layer.outChannels : filters.length,
        min: Number.isFinite(layer.min) ? layer.min : null,
        max: Number.isFinite(layer.max) ? layer.max : null,
        maxAbs: Number.isFinite(layer.maxAbs) ? layer.maxAbs : null,
        filters,
      });
    }
    return cloned;
  }

function createTrainingProfiler() {
  const stats = new Map();
  const stack = [];
  let enabled = false;
  const now =
    typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? () => performance.now()
      : () => Date.now();

  function start(name) {
    if (!enabled) return null;
    const token = { name, start: now(), children: 0 };
    stack.push(token);
    return token;
  }

  function stop(token) {
    if (!enabled || !token) return 0;
    const end = now();
    const last = stack.pop();
    if (!last || last !== token) {
      stack.length = 0;
      if (typeof console !== 'undefined' && typeof console.warn === 'function') {
        console.warn('Training profiler stack out of sync', token && token.name);
      }
      return 0;
    }
    const elapsed = end - token.start;
    const child = token.children || 0;
    const exclusive = Math.max(0, elapsed - child);
    let stat = stats.get(token.name);
    if (!stat) {
      stat = { count: 0, total: 0, self: 0, min: Infinity, max: 0 };
      stats.set(token.name, stat);
    }
    stat.count += 1;
    stat.total += elapsed;
    stat.self += exclusive;
    if (elapsed < stat.min) stat.min = elapsed;
    if (elapsed > stat.max) stat.max = elapsed;
    if (stack.length) {
      const parent = stack[stack.length - 1];
      parent.children = (parent.children || 0) + elapsed;
    }
    return elapsed;
  }

  function section(name, fn) {
    if (!enabled) return fn();
    const token = start(name);
    try {
      return fn();
    } finally {
      stop(token);
    }
  }

  function summary(options = {}) {
    const sortBy = options.sortBy || 'total';
    const descending = options.descending !== false;
    const limit = typeof options.limit === 'number' ? options.limit : null;
    const rows = [];
    for (const [name, stat] of stats.entries()) {
      const count = stat.count || 0;
      const total = stat.total || 0;
      const selfTime = stat.self || 0;
      const avg = count ? total / count : 0;
      const selfAvg = count ? selfTime / count : 0;
      rows.push({
        name,
        count,
        total,
        self: selfTime,
        average: avg,
        self_average: selfAvg,
        min: stat.min === Infinity ? 0 : stat.min,
        max: stat.max || 0,
      });
    }
    const keyMap = {
      total: (row) => row.total,
      self: (row) => row.self,
      count: (row) => row.count,
      average: (row) => row.average,
      self_average: (row) => row.self_average,
      max: (row) => row.max,
      min: (row) => row.min,
    };
    const keyFn = keyMap[sortBy] || keyMap.total;
    rows.sort((a, b) => {
      const va = keyFn(a);
      const vb = keyFn(b);
      if (va === vb) {
        return a.name.localeCompare(b.name);
      }
      return descending ? vb - va : va - vb;
    });
    return limit !== null && limit >= 0 ? rows.slice(0, limit) : rows;
  }

  function reset() {
    stats.clear();
    stack.length = 0;
  }

  function enable() {
    enabled = true;
  }

  function disable() {
    enabled = false;
    stack.length = 0;
  }

  return {
    enable,
    disable,
    reset,
    summary,
    section,
    start,
    stop,
    get enabled() {
      return enabled;
    },
  };
}

export function initTraining(game, renderer) {
  const state = game.state;
  const start = game.start;
  const updateScore = game.updateScore;
  const updateLevel = game.updateLevel;
  const recordClear = game.recordClear;
  const spawn = game.spawn;
  const log = game.log;
  const draw = renderer.draw;
  const drawNext = renderer.drawNext;

  const trainStatus = document.getElementById('train-status');
  const architectureEl = document.getElementById('model-architecture');
  const networkVizEl = document.getElementById('network-viz');
  const alphaMetricSelectEl = document.getElementById('alpha-metrics-select');
  const alphaMetricPlotsEl = document.getElementById('alpha-metric-plots');
  const alphaMetricEmptyEl = document.getElementById('alpha-metric-empty');
  const historySlider = document.getElementById('model-history-slider');
  const historyLabel = document.getElementById('model-history-label');
  const historyMeta = document.getElementById('model-history-meta');
  const mlpConfigEl = document.getElementById('mlp-config');
  const mlpHiddenCountSel = document.getElementById('mlp-hidden-count');
  const mlpLayerControlsEl = document.getElementById('mlp-layer-controls');
  const mctsSimulationInput = document.getElementById('mcts-simulations');
  const mctsCpuctInput = document.getElementById('mcts-cpuct');
  const mctsTemperatureInput = document.getElementById('mcts-temperature');

  let alphaMetricChoices = null;
  const DEFAULT_ALPHA_METRIC_NAMES = ['loss', 'policy_loss', 'value_loss'];
  const ALPHA_METRIC_LABELS = {
    loss: 'Total Loss',
    policy_loss: 'Policy Loss',
    value_loss: 'Value Loss',
  };
  const ALPHA_METRIC_STYLES = {
    loss: { stroke: 'rgba(244, 247, 121, 0.92)', fill: 'rgba(244, 247, 121, 0.18)' },
    policy_loss: { stroke: 'rgba(94, 74, 227, 0.9)', fill: 'rgba(94, 74, 227, 0.18)' },
    value_loss: { stroke: 'rgba(79, 178, 114, 0.9)', fill: 'rgba(79, 178, 114, 0.16)' },
  };
  let alphaMetricSelection = DEFAULT_ALPHA_METRIC_NAMES.slice();

  const trainingProfiler = createTrainingProfiler();
  window.__trainingProfiler = trainingProfiler;

  const LEVEL_CAP = 10;

  function logTrainingProfileSummary(limit = 6) {
    const summary = trainingProfiler.summary({ sortBy: 'total', descending: true, limit });
    const trainState = typeof window !== 'undefined' && window.__train ? window.__train : null;
    if (trainState) {
      trainState.performanceSummary = summary;
    }
    if (!summary.length) {
      log('[Perf] No training performance samples recorded.');
      return summary;
    }
    const parts = summary.map((row) =>
      `${row.name}: total=${row.total.toFixed(2)}ms self=${row.self.toFixed(2)}ms avg=${row.average.toFixed(2)}ms count=${row.count}`,
    );
    log(`[Perf] Hot sections — ${parts.join(' | ')}`);
    if (typeof console !== 'undefined') {
      if (typeof console.table === 'function') {
        const tableRows = summary.map((row) => ({
          Section: row.name,
          Count: row.count,
          'Total (ms)': Number(row.total.toFixed(2)),
          'Self (ms)': Number(row.self.toFixed(2)),
          'Avg (ms)': Number(row.average.toFixed(2)),
          'Self Avg (ms)': Number(row.self_average.toFixed(2)),
          'Min (ms)': Number(row.min.toFixed(2)),
          'Max (ms)': Number(row.max.toFixed(2)),
        }));
        console.table(tableRows);
      } else if (typeof console.log === 'function') {
        console.log('Training performance summary', summary);
      }
    }
    return summary;
  }

  window.logTrainingProfileSummary = logTrainingProfileSummary;

  if (alphaMetricSelectEl) {
    const selectedOptions = Array.isArray(alphaMetricSelectEl.selectedOptions)
      ? Array.from(alphaMetricSelectEl.selectedOptions)
      : Array.from(alphaMetricSelectEl.options || []).filter((option) => option && option.selected);
    const initialValues = selectedOptions
      .map((option) => option.value)
      .filter((value) => DEFAULT_ALPHA_METRIC_NAMES.includes(value));
    if (initialValues.length) {
      alphaMetricSelection = initialValues;
    } else {
      alphaMetricSelection = DEFAULT_ALPHA_METRIC_NAMES.slice();
      const options = Array.from(alphaMetricSelectEl.options || []);
      for (let i = 0; i < options.length; i += 1) {
        const option = options[i];
        if (option) {
          option.selected = alphaMetricSelection.includes(option.value);
        }
      }
    }
    if (typeof window !== 'undefined' && window.Choices && !alphaMetricChoices) {
      try {
        alphaMetricChoices = new window.Choices(alphaMetricSelectEl, {
          removeItemButton: true,
          shouldSort: false,
          allowHTML: false,
          itemSelectText: '',
          position: 'bottom',
        });
      } catch (err) {
        alphaMetricChoices = null;
      }
    }
    alphaMetricSelectEl.addEventListener('change', () => {
      const selected = Array.isArray(alphaMetricSelectEl.selectedOptions)
        ? Array.from(alphaMetricSelectEl.selectedOptions)
        : Array.from(alphaMetricSelectEl.options || []).filter((option) => option && option.selected);
      const values = selected
        .map((option) => option.value)
        .filter((value) => DEFAULT_ALPHA_METRIC_NAMES.includes(value));
      alphaMetricSelection = values.length ? values : [];
      const trainState = (typeof window !== 'undefined' && window.__train) ? window.__train : null;
      const alphaState = trainState && trainState.alpha ? trainState.alpha : null;
      const training = alphaState && alphaState.training ? alphaState.training : null;
      if (training && training.metricsHistory) {
        training.metricsHistory.selected = alphaMetricSelection.slice();
        renderAlphaMetricHistory(training);
      } else if (!alphaMetricSelection.length) {
        showAlphaMetricMessage('Select at least one metric to plot.');
      } else {
        showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');
      }
    });
  }

  showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');


    // Features (scaled):
    const FEATURE_NAMES = [
      'Lines',
      'Lines²',
      'Single Clear',
      'Double Clear',
      'Triple Clear',
      'Tetris',
      'Holes',
      'New Holes',
      'Bumpiness',
      'Max Height',
      'Well Sum',
      'Edge Wells',
      'Tetris Well',
      'Contact',
      'Row Transitions',
      'Col Transitions',
      'Aggregate Height',
    ];
    const FEAT_DIM = FEATURE_NAMES.length;
    const RAW_FEATURE_NAMES = (() => {
      const names = [];
      for (let r = 0; r < HEIGHT; r += 1) {
        for (let c = 0; c < WIDTH; c += 1) {
          names.push(`Cell r${r + 1}c${c + 1}`);
        }
      }
      return names;
    })();
    const RAW_FEAT_DIM = RAW_FEATURE_NAMES.length;
    const ALPHATETRIS_MODEL_LABEL = 'AlphaTetris (ConvNet)';
    const ALPHATETRIS_ARCHITECTURE_LABEL = 'ConvNet dual head policy/value network';
    const ALPHATETRIS_DEFAULT_ARCHITECTURE = `${ALPHATETRIS_MODEL_LABEL} — ${ALPHATETRIS_ARCHITECTURE_LABEL}`;
    const ALPHA_BOARD_SIZE = ALPHA_BOARD_HEIGHT * ALPHA_BOARD_WIDTH * ALPHA_BOARD_CHANNELS;
    const ALPHA_ACTION_INDEX = new Map(ALPHA_ACTIONS.map((action, index) => [`${action.rotation}|${action.column}`, index]));
    const ALPHA_TOP_OUT_VALUE = -1;
    const DEFAULT_ALPHA_BATCH_SIZE = 32;
    const DEFAULT_ALPHA_REPLAY_SIZE = 2048;
    const DEFAULT_ALPHA_LEARNING_RATE = 1e-3;
    const DEFAULT_ALPHA_VALUE_LOSS_WEIGHT = 0.5;
    const ALPHA_VIZ_UPDATE_FREQUENCY = 50;
    const ALPHA_METRIC_NAMES = DEFAULT_ALPHA_METRIC_NAMES.slice();
    const ALPHA_METRIC_HISTORY_LIMIT = 400;
    const MAX_ALPHA_SNAPSHOTS = 200;

    function sanitizePositiveInt(value, fallback) {
      if (!Number.isFinite(value) || value <= 0) {
        return fallback;
      }
      return Math.max(1, Math.floor(value));
    }

    function createAlphaTrainingState(config = {}) {
      const trainingConfig = config && typeof config.training === 'object' ? config.training : {};
      const batchSize = sanitizePositiveInt(trainingConfig.batchSize, DEFAULT_ALPHA_BATCH_SIZE);
      const replaySize = sanitizePositiveInt(trainingConfig.replaySize, DEFAULT_ALPHA_REPLAY_SIZE);
      const fitBatchSizeRaw = trainingConfig.fitBatchSize !== undefined
        ? trainingConfig.fitBatchSize
        : Math.min(batchSize, 32);
      const fitBatchSize = sanitizePositiveInt(fitBatchSizeRaw, Math.min(batchSize, 32));
      const epochs = sanitizePositiveInt(trainingConfig.epochs, 1);
      const learningRate = Number.isFinite(trainingConfig.learningRate) && trainingConfig.learningRate > 0
        ? trainingConfig.learningRate
        : DEFAULT_ALPHA_LEARNING_RATE;
      const valueLossWeight = Number.isFinite(trainingConfig.valueLossWeight)
        ? trainingConfig.valueLossWeight
        : DEFAULT_ALPHA_VALUE_LOSS_WEIGHT;
      return {
        replay: [],
        batchSize,
        maxReplaySize: replaySize,
        fitBatchSize: Math.min(batchSize, fitBatchSize),
        epochs,
        learningRate,
        valueLossWeight,
        compiledModel: null,
        optimizer: null,
        scheduled: false,
        trainingPromise: null,
        steps: 0,
        lastLoss: null,
        lastPolicyLoss: null,
        lastValueLoss: null,
        metricsHistory: { epoch: [], history: {}, metrics: [], selected: alphaMetricSelection.slice() },
        metricHistoryLimit: ALPHA_METRIC_HISTORY_LIMIT,
        nextSnapshotStep: ALPHA_VIZ_UPDATE_FREQUENCY,
        lastConvSummary: null,
        lastConvMetrics: null,
        lastConvSummaryStep: null,
      };
    }

    function ensureAlphaTrainingState(alphaState) {
      if (!alphaState) {
        return null;
      }
      if (!alphaState.training) {
        alphaState.training = createAlphaTrainingState(alphaState.config || {});
      }
      return alphaState.training;
    }

    function resetAlphaTrainingState(alphaState) {
      if (!alphaState) {
        return;
      }
      const existing = alphaState.training;
      if (existing && existing.optimizer && typeof existing.optimizer.dispose === 'function') {
        try {
          existing.optimizer.dispose();
        } catch (_) {
          /* ignore optimizer disposal failures */
        }
      }
      alphaState.training = createAlphaTrainingState(alphaState.config || {});
      alphaState.lastPreparedInputs = null;
      showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');
    }

    function prepareAlphaModelForTraining(model, alphaState) {
      if (!model) {
        return;
      }
      const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
      if (!tf || typeof model.compile !== 'function') {
        return;
      }
      const training = ensureAlphaTrainingState(alphaState);
      if (!training) {
        return;
      }
      if (training.optimizer && training.compiledModel === model) {
        return;
      }
      if (training.optimizer && typeof training.optimizer.dispose === 'function') {
        try {
          training.optimizer.dispose();
        } catch (_) {
          /* ignore optimizer disposal failures */
        }
      }
      training.optimizer = null;
      training.compiledModel = null;
      const learningRate = Number.isFinite(training.learningRate) && training.learningRate > 0
        ? training.learningRate
        : DEFAULT_ALPHA_LEARNING_RATE;
      const valueLossWeight = Number.isFinite(training.valueLossWeight)
        ? training.valueLossWeight
        : DEFAULT_ALPHA_VALUE_LOSS_WEIGHT;
      const optimizer = tf.train.adam(learningRate);
      const policyLoss = (labels, logits) => tf.tidy(() => tf.losses.softmaxCrossEntropy(labels, logits));
      const valueLoss = (labels, preds) => tf.tidy(() => tf.losses.meanSquaredError(labels, preds));
      model.compile({
        optimizer,
        loss: {
          policy: policyLoss,
          value: valueLoss,
        },
        lossWeights: { policy: 1, value: valueLossWeight },
      });
      training.optimizer = optimizer;
      training.compiledModel = model;
    }

    function clampAlphaValueTarget(value) {
      if (!Number.isFinite(value)) {
        return 0;
      }
      if (value > 1) {
        return 1;
      }
      if (value < -1) {
        return -1;
      }
      return value;
    }

    function buildAlphaPolicyTarget(policyStats, mask) {
      const target = new Float32Array(ALPHA_POLICY_SIZE);
      let sum = 0;
      if (Array.isArray(policyStats)) {
        for (let i = 0; i < policyStats.length; i += 1) {
          const entry = policyStats[i];
          if (!entry) {
            continue;
          }
          const index = ALPHA_ACTION_INDEX.get(entry.key);
          if (index === undefined) {
            continue;
          }
          const prob = Number.isFinite(entry.policy) ? entry.policy : 0;
          if (prob <= 0) {
            continue;
          }
          target[index] = prob;
          sum += prob;
        }
      }
      if (sum > 0 && Math.abs(sum - 1) > 1e-3) {
        for (let i = 0; i < target.length; i += 1) {
          target[i] /= sum;
        }
        sum = 1;
      }
      if (sum <= 0 && mask && mask.length === target.length) {
        let active = 0;
        for (let i = 0; i < mask.length; i += 1) {
          if (mask[i] > 0) {
            active += 1;
          }
        }
        if (active > 0) {
          const uniform = 1 / active;
          for (let i = 0; i < mask.length; i += 1) {
            if (mask[i] > 0) {
              target[i] = uniform;
            }
          }
          sum = 1;
        }
      }
      if (sum <= 0) {
        return null;
      }
      return target;
    }

    function sampleAlphaBatchIndices(total, count) {
      const indices = Array.from({ length: total }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
      return indices.slice(0, count);
    }

    async function runAlphaTrainingStep(alphaState) {
      const training = ensureAlphaTrainingState(alphaState);
      if (!training || !training.replay.length) {
        return;
      }
      const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
      if (!tf) {
        return;
      }
      const model = ensureAlphaModelInstance();
      if (!model || typeof model.fit !== 'function') {
        return;
      }
      prepareAlphaModelForTraining(model, alphaState);
      if (!training.optimizer) {
        return;
      }
      const total = training.replay.length;
      const batchSize = Math.min(training.batchSize, total);
      if (batchSize <= 0) {
        return;
      }
      const indices = sampleAlphaBatchIndices(total, batchSize);
      const boardBatch = new Float32Array(batchSize * ALPHA_BOARD_SIZE);
      const auxBatch = new Float32Array(batchSize * ALPHA_AUX_FEATURE_COUNT);
      const policyBatch = new Float32Array(batchSize * ALPHA_POLICY_SIZE);
      const valueBatch = new Float32Array(batchSize);
      for (let i = 0; i < batchSize; i += 1) {
        const sample = training.replay[indices[i]];
        if (!sample) {
          continue;
        }
        boardBatch.set(sample.board, i * ALPHA_BOARD_SIZE);
        auxBatch.set(sample.aux, i * ALPHA_AUX_FEATURE_COUNT);
        policyBatch.set(sample.policy, i * ALPHA_POLICY_SIZE);
        valueBatch[i] = clampAlphaValueTarget(sample.value);
      }
      const boardTensor = tf.tensor(boardBatch, [batchSize, ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, ALPHA_BOARD_CHANNELS], 'float32');
      const auxTensor = tf.tensor(auxBatch, [batchSize, ALPHA_AUX_FEATURE_COUNT], 'float32');
      const policyTensor = tf.tensor(policyBatch, [batchSize, ALPHA_POLICY_SIZE], 'float32');
      const valueTensor = tf.tensor(valueBatch, [batchSize, 1], 'float32');
      const prevSteps = training.steps;
      let epochsRan = 0;
      try {
        const fitBatchSize = Math.max(1, Math.min(training.fitBatchSize || batchSize, batchSize));
        const epochs = Math.max(1, Math.floor(training.epochs || 1));
        epochsRan = epochs;
        const history = await model.fit(
          { board: boardTensor, aux: auxTensor },
          { policy: policyTensor, value: valueTensor },
          {
            batchSize: fitBatchSize,
            epochs,
            shuffle: true,
            verbose: 0,
          },
        );
        training.steps += epochs;
        if (history && history.history) {
          const losses = history.history;
          if (Array.isArray(losses.loss) && losses.loss.length) {
            training.lastLoss = losses.loss[losses.loss.length - 1];
          }
          if (Array.isArray(losses.policy_loss) && losses.policy_loss.length) {
            training.lastPolicyLoss = losses.policy_loss[losses.policy_loss.length - 1];
          }
          if (Array.isArray(losses.value_loss) && losses.value_loss.length) {
            training.lastValueLoss = losses.value_loss[losses.value_loss.length - 1];
          }
        }
      } catch (err) {
        if (typeof console !== 'undefined' && console.error) {
          console.error('AlphaTetris training step failed', err);
        }
      } finally {
        boardTensor.dispose();
        auxTensor.dispose();
        policyTensor.dispose();
        valueTensor.dispose();
      }

      const hasMetrics = Number.isFinite(training.lastLoss)
        || Number.isFinite(training.lastPolicyLoss)
        || Number.isFinite(training.lastValueLoss);
      const metrics = hasMetrics
        ? {
            loss: Number.isFinite(training.lastLoss) ? training.lastLoss : null,
            policy_loss: Number.isFinite(training.lastPolicyLoss) ? training.lastPolicyLoss : null,
            value_loss: Number.isFinite(training.lastValueLoss) ? training.lastValueLoss : null,
          }
        : null;

      if (metrics && Number.isFinite(training.steps) && training.steps > prevSteps) {
        const appended = appendAlphaMetricHistory(training, training.steps, metrics);
        if (appended) {
          renderAlphaMetricHistory(training);
        }
        training.lastConvMetrics = metrics;
      }

      if (epochsRan > 0 && Number.isFinite(training.steps)) {
        const prevMilestones = Math.floor(prevSteps / ALPHA_VIZ_UPDATE_FREQUENCY);
        const currMilestones = Math.floor(training.steps / ALPHA_VIZ_UPDATE_FREQUENCY);
        if (currMilestones > prevMilestones) {
          const milestoneStep = currMilestones * ALPHA_VIZ_UPDATE_FREQUENCY;
          const recorded = recordAlphaMilestoneSnapshot(alphaState, training, model, milestoneStep, metrics);
          training.nextSnapshotStep = (currMilestones + 1) * ALPHA_VIZ_UPDATE_FREQUENCY;
          if (recorded) {
            updateTrainStatus();
          }
        }
      }
    }

    function scheduleAlphaTraining(alphaState) {
      const training = ensureAlphaTrainingState(alphaState);
      if (!training) {
        return;
      }
      if (!train || !train.enabled || !isAlphaModelType(train.modelType)) {
        return;
      }
      if (training.replay.length < Math.max(1, training.batchSize)) {
        return;
      }
      if (training.trainingPromise || training.scheduled) {
        return;
      }
      training.scheduled = true;
      const scheduleFn = () => {
        training.scheduled = false;
        training.trainingPromise = runAlphaTrainingStep(alphaState)
          .catch((err) => {
            if (typeof console !== 'undefined' && console.error) {
              console.error('AlphaTetris training promise rejected', err);
            }
          })
          .finally(() => {
            training.trainingPromise = null;
            if (train && train.enabled && isAlphaModelType(train.modelType)
              && training.replay.length >= Math.max(1, training.batchSize)) {
              scheduleAlphaTraining(alphaState);
            }
          });
      };
      if (typeof queueMicrotask === 'function') {
        queueMicrotask(scheduleFn);
      } else {
        Promise.resolve().then(scheduleFn);
      }
    }

    function queueAlphaTrainingExample(alphaState, example) {
      if (!alphaState || !example || !example.board || !example.aux || !example.policy) {
        return;
      }
      const training = ensureAlphaTrainingState(alphaState);
      if (!training) {
        return;
      }
      const stored = {
        board: example.board instanceof Float32Array ? example.board : new Float32Array(example.board),
        aux: example.aux instanceof Float32Array ? example.aux : new Float32Array(example.aux),
        policy: example.policy instanceof Float32Array ? example.policy : new Float32Array(example.policy),
        value: clampAlphaValueTarget(example.value),
      };
      training.replay.push(stored);
      if (training.replay.length > training.maxReplaySize) {
        training.replay.splice(0, training.replay.length - training.maxReplaySize);
      }
      if (train && train.enabled && isAlphaModelType(train.modelType)) {
        scheduleAlphaTraining(alphaState);
      }
    }

    function recordAlphaTrainingExample(policyStats, rootValue) {
      if (!train || !train.enabled || !isAlphaModelType(train.modelType)) {
        const alphaState = train && train.alpha ? train.alpha : null;
        if (alphaState) {
          alphaState.lastPreparedInputs = null;
        }
        return;
      }
      const alphaState = ensureAlphaState();
      const prepared = alphaState && alphaState.lastPreparedInputs ? alphaState.lastPreparedInputs : null;
      if (!prepared || !prepared.board || !prepared.aux) {
        return;
      }
      const policyTarget = buildAlphaPolicyTarget(policyStats, prepared.mask || null);
      if (!policyTarget) {
        alphaState.lastPreparedInputs = null;
        return;
      }
      const sample = {
        board: new Float32Array(prepared.board),
        aux: new Float32Array(prepared.aux),
        policy: policyTarget,
        value: clampAlphaValueTarget(rootValue),
      };
      alphaState.lastPreparedInputs = null;
      queueAlphaTrainingExample(alphaState, sample);
    }

    function isAlphaModelType(type) {
      return type === 'alphatetris';
    }
    function isMlpModelType(type) {
      return type === 'mlp' || type === 'mlp_raw';
    }
    function usesPopulationModel(type) {
      return !isAlphaModelType(type);
    }
    function resolveMlpType(type) {
      return type === 'mlp_raw' ? 'mlp_raw' : 'mlp';
    }
    function inputDimForModel(type) {
      if (isAlphaModelType(type)) {
        return null;
      }
      return type === 'mlp_raw' ? RAW_FEAT_DIM : FEAT_DIM;
    }
    function featureNamesForModel(type) {
      if (isAlphaModelType(type)) {
        return null;
      }
      return type === 'mlp_raw' ? RAW_FEATURE_NAMES : FEATURE_NAMES;
    }
    function modelDisplayName(type) {
      if (type === 'mlp_raw') {
        return 'MLP (board occupancy)';
      }
      if (type === 'mlp') {
        return 'MLP (engineered features)';
      }
      if (isAlphaModelType(type)) {
        return ALPHATETRIS_MODEL_LABEL;
      }
      return 'Linear';
    }
    function normalizeAlphaArchitectureDescription(text) {
      if (typeof text !== 'string') {
        return '';
      }
      const trimmed = text.trim();
      if (!trimmed) {
        return '';
      }
      const lower = trimmed.toLowerCase();
      if (lower.includes('alphatetris')) {
        let normalized = trimmed;
        if (lower.includes('alphatetris convnet') && !lower.includes('alphatetris (convnet)')) {
          normalized = normalized.replace(/alphatetris\s*convnet/gi, ALPHATETRIS_MODEL_LABEL);
        }
        if (normalized.trim().toLowerCase() === ALPHATETRIS_MODEL_LABEL.toLowerCase()) {
          return ALPHATETRIS_DEFAULT_ARCHITECTURE;
        }
        return normalized;
      }
      return `${ALPHATETRIS_MODEL_LABEL} — ${trimmed}`;
    }
    const AI_STEP_MS = 28; // ms between AI animation steps

    const ROW_MASK_LIMIT = (1 << WIDTH) - 1;
    const rowTransitionTable =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(ROW_MASK_LIMIT + 1) : new Array(ROW_MASK_LIMIT + 1).fill(0);
    const rowPopcountTable =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(ROW_MASK_LIMIT + 1) : new Array(ROW_MASK_LIMIT + 1).fill(0);
    const rowHorizontalContactTable =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(ROW_MASK_LIMIT + 1) : new Array(ROW_MASK_LIMIT + 1).fill(0);
    const bitIndexTable = (() => {
      const size = ROW_MASK_LIMIT + 1;
      const table = typeof Int16Array !== 'undefined' ? new Int16Array(size) : new Array(size);
      if (typeof table.fill === 'function') {
        table.fill(-1);
      } else {
        for (let i = 0; i < size; i += 1) table[i] = -1;
      }
      for (let c = 0; c < WIDTH; c += 1) {
        table[1 << c] = c;
      }
      return table;
    })();

    function countBits32(value) {
      value -= (value >> 1) & 0x55555555;
      value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
      return (((value + (value >> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }

    (function initRowMetricTables() {
      for (let mask = 0; mask <= ROW_MASK_LIMIT; mask += 1) {
        let prev = 0;
        let transitions = 0;
        for (let c = 0; c < WIDTH; c += 1) {
          const filled = (mask >> c) & 1;
          if (filled !== prev) {
            transitions += 1;
            prev = filled;
          }
        }
        if (prev !== 0) {
          transitions += 1;
        }
        rowTransitionTable[mask] = transitions;
        rowPopcountTable[mask] = countBits32(mask);
        const horizontalPairs = countBits32(mask & ((mask << 1) & ROW_MASK_LIMIT));
        rowHorizontalContactTable[mask] = horizontalPairs * 2;
      }
    })();

    // MLP architecture (when selected)
    const DEFAULT_MLP_HIDDEN = [8];
    const MLP_MIN_HIDDEN_LAYERS = 1;
    const MLP_MAX_HIDDEN_LAYERS = 3;
    const MLP_MIN_UNITS = 1;
    const MLP_MAX_UNITS = 32;
    let mlpHiddenLayers = DEFAULT_MLP_HIDDEN.slice();
    let currentModelType = 'linear';

    function sanitizeHiddenUnits(value, idx, fallback){
      const fallbackSource = Number.isFinite(fallback)
        ? Math.round(fallback)
        : (DEFAULT_MLP_HIDDEN[Math.min(idx, DEFAULT_MLP_HIDDEN.length - 1)] ?? DEFAULT_MLP_HIDDEN[0] ?? 8);
      const n = Number(value);
      if(!Number.isFinite(n)){
        return Math.max(MLP_MIN_UNITS, Math.min(MLP_MAX_UNITS, fallbackSource));
      }
      const rounded = Math.round(n);
      return Math.max(MLP_MIN_UNITS, Math.min(MLP_MAX_UNITS, rounded));
    }

    function sanitizeSimulationCount(value, fallback){
      const raw = Math.round(Number(value));
      if(!Number.isFinite(raw) || raw <= 0){
        const fb = Math.round(Number(fallback) || 1);
        return Math.max(1, Math.min(MAX_AI_STEPS_PER_FRAME, fb || 1));
      }
      return Math.max(1, Math.min(MAX_AI_STEPS_PER_FRAME, raw));
    }

    function sanitizeExplorationConstant(value, fallback){
      const parsed = Number(value);
      if(!Number.isFinite(parsed)){
        return Number.isFinite(fallback) ? fallback : 1.5;
      }
      return Math.max(0.01, parsed);
    }

    function sanitizeTemperature(value, fallback){
      const parsed = Number(value);
      if(!Number.isFinite(parsed) || parsed < 0){
        return Math.max(0, Number.isFinite(fallback) ? fallback : 1);
      }
      return Math.max(0, parsed);
    }

    function mlpParamDim(layers = mlpHiddenLayers, modelType = currentModelType){
      const resolvedType = resolveMlpType(modelType);
      const inputDim = inputDimForModel(resolvedType);
      const count = Math.max(MLP_MIN_HIDDEN_LAYERS, Math.min(MLP_MAX_HIDDEN_LAYERS, layers.length || 0));
      let dim = 0;
      let prev = inputDim;
      for(let i = 0; i < count; i++){
        const size = sanitizeHiddenUnits(layers[i], i, layers[i]);
        dim += prev * size + size;
        prev = size;
      }
      dim += prev + 1; // output layer (1 unit + bias)
      return dim;
    }

    function currentMlpLayerSizes(modelType = currentModelType, layers = mlpHiddenLayers){
      const resolvedType = resolveMlpType(modelType);
      const count = Math.max(MLP_MIN_HIDDEN_LAYERS, Math.min(MLP_MAX_HIDDEN_LAYERS, layers.length || 0));
      const sizes = [];
      for(let i = 0; i < count; i++){
        sizes.push(sanitizeHiddenUnits(layers[i], i, layers[i]));
      }
      const inputDim = inputDimForModel(resolvedType);
      return [inputDim, ...sizes, 1];
    }

    // Numeric dtype for weight arrays
    const HAS_F16 = (typeof Float16Array !== 'undefined');
    const DEFAULT_DTYPE = HAS_F16 ? 'f16' : 'f32';
    const SCORE_PLOT_DEFAULT_UPDATE_FREQ = 5;
    const SCORE_PLOT_ALPHATETRIS_UPDATE_FREQ = 20;
    let dtypePreference = DEFAULT_DTYPE;
    function allocFloat32(n){ return new Float32Array(n); }
    function copyValues(source, target){
      const src = source || [];
      const srcLength = (typeof src.length === 'number') ? src.length : 0;
      const limit = Math.min(target.length, srcLength);
      for(let i = 0; i < limit; i++){
        const value = src[i];
        target[i] = Number.isFinite(value) ? value : 0;
      }
      for(let i = limit; i < target.length; i++){
        target[i] = 0;
      }
    }
    function createFloat32ArrayFrom(values){
      if(!values || !values.length){
        return new Float32Array(0);
      }
      const arr = allocFloat32(values.length);
      copyValues(values, arr);
      return arr;
    }
    function createDisplayView(master, existing){
      if(!master){
        return null;
      }
      const activeModelType = train ? train.modelType : currentModelType;
      if(isAlphaModelType(activeModelType)){
        return master;
      }
      if(dtypePreference !== 'f16' || !HAS_F16){
        return master;
      }
      const view = existing && existing.length === master.length ? existing : new Float16Array(master.length);
      copyValues(master, view);
      return view;
    }

    // Initial weights for Linear model (intentionally poor to make the very first attempt worse)
    // Order: [lines, lines2, is1, is2, is3, is4, holes, newHoles, bumpiness, maxH, wellSum, edgeWell, tetrisWell, contact, rowTrans, colTrans, aggH]
    const INITIAL_MEAN_LINEAR_BASE = [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.4, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2];
    const INITIAL_STD_LINEAR_BASE  = new Array(FEAT_DIM).fill(0.4);

    function paramDim(model = currentModelType, layers = mlpHiddenLayers){
      const targetModel = model || currentModelType;
      if(!usesPopulationModel(targetModel)){
        return 0;
      }
      if(isMlpModelType(targetModel)){
        const layerConfig = Array.isArray(layers) ? layers : mlpHiddenLayers;
        return mlpParamDim(layerConfig, targetModel);
      }
      return FEAT_DIM;
    }
    function makeStatsArray(vals){ return createFloat32ArrayFrom(vals); }
    function cloneWeightsArray(source){
      if(!source || !source.length){
        return new Float64Array(0);
      }
      const copy = new Float64Array(source.length);
      for(let i = 0; i < source.length; i++){
        copy[i] = source[i];
      }
      return copy;
    }
    function initialMean(model, layers = mlpHiddenLayers){
      if(isAlphaModelType(model)){
        return makeStatsArray([]);
      }
      if(isMlpModelType(model)){
        const dim = mlpParamDim(layers, model);
        const base = new Array(dim).fill(0.0);
        return makeStatsArray(base);
      }
      return makeStatsArray(INITIAL_MEAN_LINEAR_BASE);
    }
    function initialStd(model, layers = mlpHiddenLayers){
      if(isAlphaModelType(model)){
        return makeStatsArray([]);
      }
      if(isMlpModelType(model)){
        const dim = mlpParamDim(layers, model);
        const base = new Array(dim).fill(0.2);
        return makeStatsArray(base);
      }
      return makeStatsArray(INITIAL_STD_LINEAR_BASE);
    }

    function createAlphaState(prevConfig = null){
      const baseConfig = prevConfig && typeof prevConfig === 'object' ? { ...prevConfig } : {};
      const existingDescription =
        typeof baseConfig.architectureDescription === 'string' ? baseConfig.architectureDescription : '';
      const normalizedExisting = normalizeAlphaArchitectureDescription(existingDescription);
      if(normalizedExisting){
        baseConfig.architectureDescription = normalizedExisting;
      } else {
        const fallbackSource =
          typeof baseConfig.description === 'string' && baseConfig.description.trim()
            ? baseConfig.description.trim()
            : ALPHATETRIS_ARCHITECTURE_LABEL;
        const normalizedFallback = normalizeAlphaArchitectureDescription(fallbackSource);
        baseConfig.architectureDescription = normalizedFallback || ALPHATETRIS_DEFAULT_ARCHITECTURE;
      }
      return {
        config: baseConfig,
        modelPromise: null,
        latestModel: null,
        lastRootValue: 0,
        lastPolicyLogits: null,
      };
    }

    function ensureAlphaState(){
      if(!train.alpha){
        train.alpha = createAlphaState();
      }
      return train.alpha;
    }

    function disposeAlphaModel(alphaState){
      if(!alphaState || !alphaState.latestModel){
        return;
      }
      try {
        if(typeof alphaState.latestModel.dispose === 'function'){
          alphaState.latestModel.dispose();
        }
      } catch (err) {
        if(typeof console !== 'undefined' && console.warn){
          console.warn('Failed to dispose AlphaTetris model', err);
        }
      }
      alphaState.latestModel = null;
      alphaState.modelPromise = null;
    }

    function ensureAlphaModelInstance(){
      const alphaState = ensureAlphaState();
      if(alphaState.latestModel){
        return alphaState.latestModel;
      }
      try {
        const model = buildAlphaTetrisModel(alphaState.config || {});
        alphaState.latestModel = model;
        alphaState.modelPromise = Promise.resolve(model);
        return model;
      } catch (err) {
        if(typeof console !== 'undefined' && console.error){
          console.error('Failed to build AlphaTetris model', err);
        }
        alphaState.modelPromise = Promise.reject(err);
        return null;
      }
    }

    function describeModelArchitecture(){
      if(isAlphaModelType(currentModelType)){
        return describeAlphaArchitecture();
      }
      if(isMlpModelType(currentModelType)){
        const sizes = currentMlpLayerSizes(currentModelType);
        if(!sizes.length) return 'Architecture: unavailable';
        const parts = sizes.map((size, idx) => {
          if(idx === 0) return `${size} in`;
          if(idx === sizes.length - 1) return `${size} out`;
          return `${size}`;
        });
        const descriptor = currentModelType === 'mlp_raw' ? 'Raw board MLP' : 'MLP';
        return `Architecture: ${descriptor} — ${parts.join(' → ')}`;
      }
      return `Linear policy with ${FEAT_DIM} inputs`;
    }

    function describeAlphaArchitecture(){
      const trainState = (typeof window !== 'undefined' && window.__train) ? window.__train : null;
      const alphaState = trainState && trainState.alpha ? trainState.alpha : null;
      const config = alphaState && alphaState.config ? alphaState.config : null;
      if(config){
        const description =
          typeof config.architectureDescription === 'string' && config.architectureDescription.trim()
            ? config.architectureDescription.trim()
            : (typeof config.description === 'string' && config.description.trim()
                ? config.description.trim()
                : '');
        if(description){
          const normalized = normalizeAlphaArchitectureDescription(description) || ALPHATETRIS_DEFAULT_ARCHITECTURE;
          return `Architecture: ${normalized}`;
        }
        if(Array.isArray(config.layers) && config.layers.length){
          const parts = config.layers.map((layer) => String(layer));
          return `Architecture: ${ALPHATETRIS_MODEL_LABEL} — ${parts.join(' → ')}`;
        }
      }
      return `Architecture: ${ALPHATETRIS_DEFAULT_ARCHITECTURE}`;
    }

    function describeSnapshotArchitecture(entry){
      if(!entry){
        return describeModelArchitecture();
      }
      const genLabel = (() => {
        if(isAlphaModelType(entry.modelType) && Number.isFinite(entry.step)){
          return `Step ${entry.step}`;
        }
        if(Number.isFinite(entry.gen)){
          return `Gen ${entry.gen}`;
        }
        return 'Saved model';
      })();
      const layerSizes = Array.isArray(entry.layerSizes) ? entry.layerSizes : null;
      if(isMlpModelType(entry.modelType) && layerSizes && layerSizes.length >= 2){
        const parts = layerSizes.map((size, idx) => {
          if(idx === 0) return `${size} in`;
          if(idx === layerSizes.length - 1) return `${size} out`;
          return `${size}`;
        });
        const descriptor = entry.modelType === 'mlp_raw' ? 'Raw board MLP' : 'MLP';
        return `${genLabel} — ${descriptor}: ${parts.join(' → ')}`;
      }
      if(entry.modelType === 'linear'){
        const inputCount = layerSizes && layerSizes.length ? layerSizes[0] : FEAT_DIM;
        return `${genLabel} — Linear policy with ${inputCount} inputs`;
      }
      if(isAlphaModelType(entry.modelType)){
        const rawAlphaDesc =
          (entry && typeof entry.architectureDescription === 'string' && entry.architectureDescription.trim())
            ? entry.architectureDescription.trim()
            : (entry && typeof entry.alphaDescription === 'string' && entry.alphaDescription.trim()
                ? entry.alphaDescription.trim()
                : ALPHATETRIS_DEFAULT_ARCHITECTURE);
        const alphaDesc = normalizeAlphaArchitectureDescription(rawAlphaDesc) || ALPHATETRIS_DEFAULT_ARCHITECTURE;
        return `${genLabel} — ${alphaDesc}`;
      }
      return `${genLabel} — Architecture unavailable`;
    }

    function formatScore(value){
      if(!Number.isFinite(value)){
        return 'n/a';
      }
      return Math.round(value).toLocaleString();
    }

    function formatAlphaSnapshotMetrics(entry){
      if(!entry || entry.modelType !== 'alphatetris'){
        return [];
      }
      const metrics = entry.metrics || {};
      const parts = [];
      const loss = metrics.loss;
      const policyLoss = metrics.policy_loss;
      const valueLoss = metrics.value_loss;
      if(Number.isFinite(loss)){
        parts.push(`Loss ${loss.toFixed(4)}`);
      }
      if(Number.isFinite(policyLoss)){
        parts.push(`Policy ${policyLoss.toFixed(4)}`);
      }
      if(Number.isFinite(valueLoss)){
        parts.push(`Value ${valueLoss.toFixed(4)}`);
      }
      return parts;
    }

    function getHistorySelection(){
      if(!train || !Array.isArray(train.bestByGeneration) || !train.bestByGeneration.length){
        return null;
      }
      if(train.historySelection === null || train.historySelection < 0){
        return null;
      }
      const capped = Math.min(train.bestByGeneration.length - 1, train.historySelection);
      if(capped < 0){
        return null;
      }
      return { entry: train.bestByGeneration[capped], index: capped };
    }

    function syncHistoryControls(){
      if(!historySlider) return;
      const total = train && Array.isArray(train.bestByGeneration) ? train.bestByGeneration.length : 0;
      const hasHistory = total > 0;
      historySlider.min = 0;
      const sliderMax = hasHistory ? total : 0;
      historySlider.max = sliderMax;
      historySlider.step = 1;
      const selection = train && train.historySelection !== null ? Math.max(0, Math.min(total - 1, train.historySelection)) : null;
      const sliderValue = hasHistory ? (selection !== null ? selection : sliderMax) : 0;
      historySlider.value = String(sliderValue);
      historySlider.disabled = !hasHistory;
      if(hasHistory){
        const fill = sliderMax > 0 ? Math.max(0, Math.min(100, (sliderValue / sliderMax) * 100)) : 0;
        const active = 'rgba(141, 225, 173, 0.85)';
        const base = 'rgba(94, 74, 227, 0.25)';
        historySlider.style.background = `linear-gradient(90deg, ${active} 0%, ${active} ${fill}%, ${base} ${fill}%, ${base} 100%)`;
      } else {
        historySlider.style.background = 'linear-gradient(90deg, rgba(94, 74, 227, 0.35), rgba(141, 225, 173, 0.55))';
      }

      const activeIndex = (!hasHistory || sliderValue >= sliderMax)
        ? null
        : Math.max(0, Math.min(total - 1, Math.round(sliderValue)));

      if(historyLabel){
        if(activeIndex === null){
          historyLabel.textContent = 'Live (current training)';
        } else {
          const entry = train.bestByGeneration[activeIndex];
          if(entry && isAlphaModelType(entry.modelType) && Number.isFinite(entry.step)){
            historyLabel.textContent = `Step ${entry.step}`;
          } else if(entry && Number.isFinite(entry.gen)){
            historyLabel.textContent = `Gen ${entry.gen}`;
          } else {
            historyLabel.textContent = 'Stored snapshot';
          }
        }
      }

      if(historyMeta){
        if(!hasHistory){
          historyMeta.textContent = 'Best-of-generation snapshots will appear as training progresses.';
        } else if(activeIndex === null){
          const latest = train.bestByGeneration[total - 1];
          const info = [];
          if(latest && latest.modelType === 'alphatetris'){
            if(Number.isFinite(latest.step)) info.push(`Latest stored: Step ${latest.step}`);
            const metricsParts = formatAlphaSnapshotMetrics(latest);
            if(metricsParts.length){
              info.push(...metricsParts);
            }
            if(!metricsParts.length && latest.modelType){
              info.push(modelDisplayName(latest.modelType));
            }
          } else {
            if(Number.isFinite(latest?.gen)) info.push(`Latest stored: Gen ${latest.gen}`);
            if(Number.isFinite(latest?.fitness)) info.push(`Score ${formatScore(latest.fitness)}`);
            if(latest?.modelType) info.push(modelDisplayName(latest.modelType));
          }
          historyMeta.textContent = info.length ? info.join(' • ') : 'Snapshot details unavailable.';
        } else {
          const entry = train.bestByGeneration[activeIndex];
          const info = [];
          if(entry && entry.modelType === 'alphatetris'){
            if(Number.isFinite(entry.step)) info.push(`Step ${entry.step}`);
            const metricsParts = formatAlphaSnapshotMetrics(entry);
            if(metricsParts.length){
              info.push(...metricsParts);
            } else if(entry.modelType){
              info.push(modelDisplayName(entry.modelType));
            }
          } else {
            if(entry?.modelType) info.push(modelDisplayName(entry.modelType));
            if(Number.isFinite(entry?.fitness)) info.push(`Score ${formatScore(entry.fitness)}`);
          }
          historyMeta.textContent = info.length ? info.join(' • ') : 'Snapshot details unavailable.';
        }
      }
    }

    function syncMctsControls(){
      if(!train || !train.ai || !train.ai.search){
        return;
      }
      const search = train.ai.search;
      if(mctsSimulationInput){
        const sanitized = sanitizeSimulationCount(search.simulations, search.simulations);
        if(search.simulations !== sanitized){
          search.simulations = sanitized;
        }
        if(mctsSimulationInput.value !== String(sanitized)){
          mctsSimulationInput.value = String(sanitized);
        }
      }
      if(mctsCpuctInput){
        const sanitized = sanitizeExplorationConstant(search.cPuct, search.cPuct);
        if(search.cPuct !== sanitized){
          search.cPuct = sanitized;
        }
        if(mctsCpuctInput.value !== String(sanitized)){
          mctsCpuctInput.value = String(sanitized);
        }
      }
      if(mctsTemperatureInput){
        const sanitized = sanitizeTemperature(search.temperature, search.temperature);
        if(search.temperature !== sanitized){
          search.temperature = sanitized;
        }
        if(mctsTemperatureInput.value !== String(sanitized)){
          mctsTemperatureInput.value = String(sanitized);
        }
      }
    }

    function recordGenerationSnapshot(snapshot){
      if(!train) return;
      if(!Array.isArray(train.bestByGeneration)){
        train.bestByGeneration = [];
      }
      const prevLength = train.bestByGeneration.length;
      train.bestByGeneration.push({
        gen: snapshot.gen,
        step: Number.isFinite(snapshot.step) ? snapshot.step : (Number.isFinite(snapshot.gen) ? snapshot.gen : null),
        fitness: snapshot.fitness,
        modelType: snapshot.modelType,
        dtype: snapshot.dtype || train.dtype || DEFAULT_DTYPE,
        layerSizes: Array.isArray(snapshot.layerSizes) ? snapshot.layerSizes.slice() : null,
        weights: snapshot.weights,
        scoreIndex: Number.isFinite(snapshot.scoreIndex) ? snapshot.scoreIndex : null,
        recordedAt: snapshot.recordedAt || Date.now(),
        metrics: snapshot.metrics ? { ...snapshot.metrics } : null,
        convSummary: snapshot.convSummary ? cloneAlphaConvSummary(snapshot.convSummary) : null,
        architectureDescription: snapshot.architectureDescription
          || snapshot.alphaDescription
          || snapshot.architecture
          || null,
      });
      if(train.historySelection !== null){
        const lastIdxBefore = Math.max(0, prevLength - 1);
        if(prevLength === 0 || train.historySelection >= lastIdxBefore){
          train.historySelection = train.bestByGeneration.length - 1;
        }
      }
      syncHistoryControls();
    }

    function activeWeightArray(){
      if(train && isAlphaModelType(train.modelType)){
        return null;
      }
      if(train && train.currentWeightsOverride && train.currentWeightsOverride.length){
        return train.currentWeightsOverride;
      }
      if(train && train.enabled && train.candIndex >= 0 && train.candIndex < train.candWeights.length){
        const candidate = train.candWeights[train.candIndex];
        if(candidate && candidate.length){
          return candidate;
        }
      }
      if(train && train.mean && train.mean.length){
        return train.mean;
      }
      if(train && train.bestEverWeights && train.bestEverWeights.length){
        return train.bestEverWeights;
      }
      return null;
    }

    async function createWeightSnapshot(){
      if(train && isAlphaModelType(train.modelType)){
        try {
          const alphaState = ensureAlphaState();
          const model = alphaState.latestModel || ensureAlphaModelInstance();
          if(!model){
            log('AlphaTetris model is not ready to export yet.');
            return null;
          }
          const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
          if(!tf){
            log('TensorFlow.js is unavailable. Cannot export AlphaTetris model.');
            return null;
          }
          let artifacts = null;
          await model.save(tf.io.withSaveHandler(async (modelArtifacts) => {
            artifacts = {
              modelTopology: modelArtifacts.modelTopology,
              weightSpecs: modelArtifacts.weightSpecs,
              weightDataBase64: arrayBufferToBase64(modelArtifacts.weightData),
            };
            return {
              modelArtifactsInfo: {
                dateSaved: new Date(),
                modelTopologyType: 'JSON',
                modelTopologyBytes: modelArtifacts.modelTopology
                  ? JSON.stringify(modelArtifacts.modelTopology).length
                  : 0,
                weightSpecsBytes: modelArtifacts.weightSpecs
                  ? JSON.stringify(modelArtifacts.weightSpecs).length
                  : 0,
                weightDataBytes: modelArtifacts.weightData ? modelArtifacts.weightData.byteLength : 0,
              },
            };
          }));
          if(!artifacts){
            throw new Error('Failed to capture AlphaTetris model artifacts.');
          }
          const architectureDescription = alphaState && alphaState.config && alphaState.config.architectureDescription
            ? alphaState.config.architectureDescription
            : ALPHATETRIS_DEFAULT_ARCHITECTURE;
          return {
            version: 1,
            createdAt: new Date().toISOString(),
            modelType: 'alphatetris',
            dtype: 'f32',
            architectureDescription,
            alphaModel: artifacts,
          };
        } catch (err) {
          console.error(err);
          log('Failed to export AlphaTetris model.');
          return null;
        }
      }
      const weights = activeWeightArray();
      if(!weights || !weights.length){
        return null;
      }
      const values = Array.from(weights, (v) => Number(v));
      const activeModelType = train ? train.modelType : currentModelType;
      const hiddenLayersSnapshot = (train && isMlpModelType(activeModelType))
        ? (() => {
            const sizes = currentMlpLayerSizes(activeModelType);
            if(!sizes || sizes.length <= 2){
              return [];
            }
            return sizes.slice(1, sizes.length - 1).map((size, idx) => sanitizeHiddenUnits(size, idx, size));
          })()
        : [];
      const snapshot = {
        version: 1,
        createdAt: new Date().toISOString(),
        modelType: activeModelType,
        dtype: (train && train.dtype) ? train.dtype : DEFAULT_DTYPE,
        featureCount: inputDimForModel(activeModelType),
        hiddenLayers: hiddenLayersSnapshot,
        weights: values,
      };
      if(train && Number.isFinite(train.gen)){
        snapshot.generation = train.gen;
      }
      if(train && Number.isFinite(train.bestEverFitness)){
        snapshot.bestEverFitness = train.bestEverFitness;
      }
      return snapshot;
    }

    async function downloadCurrentWeights(){
      try {
        const snapshot = await createWeightSnapshot();
        if(!snapshot){
          log('Weights unavailable for download yet. Start a game or training session first.');
          return;
        }
        const text = JSON.stringify(snapshot, null, 2);
        const blob = new Blob([text], { type: 'application/json' });
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const safeModel = (snapshot.modelType || 'model').toLowerCase();
        const fileName = `tetris-${safeModel}-weights-${timestamp}.json`;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        setTimeout(() => {
          try {
            URL.revokeObjectURL(link.href);
          } catch (_) {}
          if(link.parentNode){
            link.parentNode.removeChild(link);
          }
        }, 0);
        const isAlpha = isAlphaModelType(snapshot.modelType);
        const paramCount = isAlpha
          ? countWeightsFromSpecs(snapshot.alphaModel && snapshot.alphaModel.weightSpecs)
          : (Array.isArray(snapshot.weights) ? snapshot.weights.length : 0);
        const formattedCount = Number.isFinite(paramCount) ? paramCount.toLocaleString() : 'unknown';
        const noun = isAlpha ? 'model' : 'weights';
        log(`Saved ${modelDisplayName(snapshot.modelType)} ${noun} (${formattedCount} params) to ${fileName}.`);
      } catch (err) {
        console.error(err);
        const message = (err && err.message) ? err.message : 'unknown error';
        log(`Failed to export weights: ${message}`);
      }
    }

    function parseWeightSnapshot(text){
      if(!text || !text.trim()){
        throw new Error('File was empty');
      }
      let data;
      try {
        data = JSON.parse(text);
      } catch (err) {
        throw new Error('File was not valid JSON');
      }
      if(!data || typeof data !== 'object'){
        throw new Error('Snapshot data missing');
      }
      if(data.version && Number(data.version) !== 1){
        throw new Error(`Unsupported snapshot version ${data.version}`);
      }
      const modelType = data.modelType;
      if(!modelType){
        throw new Error('Snapshot missing model type');
      }
      if(isAlphaModelType(modelType)){
        const alpha = data.alphaModel;
        if(!alpha || typeof alpha !== 'object'){
          throw new Error('AlphaTetris snapshot missing model artifacts');
        }
        const topology = alpha.modelTopology;
        const weightSpecs = Array.isArray(alpha.weightSpecs) ? alpha.weightSpecs : null;
        const base64 = typeof alpha.weightDataBase64 === 'string' && alpha.weightDataBase64
          ? alpha.weightDataBase64
          : null;
        const fallbackBuffer = alpha.weightData instanceof ArrayBuffer ? alpha.weightData : null;
        const weightData = base64 ? base64ToArrayBuffer(base64) : fallbackBuffer;
        if(!topology){
          throw new Error('AlphaTetris snapshot missing model topology');
        }
        if(!weightSpecs || !weightSpecs.length){
          throw new Error('AlphaTetris snapshot missing weight specs');
        }
        if(!weightData || !weightData.byteLength){
          throw new Error('AlphaTetris snapshot missing weight data');
        }
        const descriptionSource = typeof data.architectureDescription === 'string'
          ? data.architectureDescription
          : (typeof data.alphaDescription === 'string' ? data.alphaDescription : '');
        const normalizedDescription = normalizeAlphaArchitectureDescription(descriptionSource)
          || ALPHATETRIS_DEFAULT_ARCHITECTURE;
        data.alphaModel = {
          modelTopology: topology,
          weightSpecs,
          weightData,
          weightDataBase64: base64 || arrayBufferToBase64(weightData),
        };
        data.architectureDescription = normalizedDescription;
        data.dtype = 'f32';
        data.version = 1;
        return data;
      }
      if(modelType !== 'linear' && !isMlpModelType(modelType)){
        throw new Error('Snapshot missing model type');
      }
      if(!Array.isArray(data.weights) || !data.weights.length){
        throw new Error('Snapshot missing weights');
      }
      const expectedFeatures = inputDimForModel(modelType || 'linear');
      if(data.featureCount && data.featureCount !== expectedFeatures){
        throw new Error(`Snapshot expects ${data.featureCount} features but this build uses ${expectedFeatures}`);
      }
      data.version = 1;
      return data;
    }

    async function applyWeightSnapshot(snapshot, context){
      if(!snapshot){
        throw new Error('Snapshot missing');
      }
      const modelType = snapshot.modelType;
      let dtype = snapshot.dtype;
      if(dtype === 'f16' && !HAS_F16){
        dtype = 'f32';
      }
      if(dtype !== 'f16' && dtype !== 'f32'){
        dtype = 'f32';
      }

      const wasRunning = train && train.enabled;
      if(wasRunning){
        stopTraining();
      }

      if(modelSel){
        modelSel.value = modelType;
      }

      if(isAlphaModelType(modelType)){
        const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
        if(!tf){
          throw new Error('TensorFlow.js is unavailable. Cannot load AlphaTetris model.');
        }
        const alphaArtifacts = snapshot.alphaModel || {};
        const modelTopology = alphaArtifacts.modelTopology;
        const weightSpecs = Array.isArray(alphaArtifacts.weightSpecs) ? alphaArtifacts.weightSpecs : null;
        const weightData = alphaArtifacts.weightData instanceof ArrayBuffer
          ? alphaArtifacts.weightData
          : (typeof alphaArtifacts.weightDataBase64 === 'string'
            ? base64ToArrayBuffer(alphaArtifacts.weightDataBase64)
            : null);
        if(!modelTopology || !weightSpecs || !weightSpecs.length || !weightData || !weightData.byteLength){
          throw new Error('AlphaTetris snapshot was missing model parameters.');
        }
        const normalizedDescription = normalizeAlphaArchitectureDescription(snapshot.architectureDescription)
          || ALPHATETRIS_DEFAULT_ARCHITECTURE;
        if(train.alpha){
          disposeAlphaModel(train.alpha);
        }
        train.alpha = createAlphaState({ architectureDescription: normalizedDescription });
        train.modelType = modelType;
        currentModelType = modelType;
        train.dtype = 'f32';
        dtypePreference = 'f32';
        resetTraining();
        syncMlpConfigVisibility();
        const alphaState = ensureAlphaState();
        alphaState.config = alphaState.config || {};
        alphaState.config.architectureDescription = normalizedDescription;
        alphaState.config.description = normalizedDescription;
        const handler = tf.io.fromMemory(modelTopology, weightSpecs, weightData);
        const loadedModel = await tf.loadLayersModel(handler);
        alphaState.latestModel = loadedModel;
        alphaState.modelPromise = Promise.resolve(loadedModel);
        alphaState.lastPolicyLogits = null;
        alphaState.lastRootValue = 0;
        train.bestEverFitness = Number.isFinite(snapshot.bestEverFitness)
          ? snapshot.bestEverFitness
          : -Infinity;
        const snapGen = Number(snapshot.generation);
        if(Number.isFinite(snapGen) && snapGen >= 0){
          train.gen = snapGen;
        }
        updateTrainStatus();
        const origin = context && context.fileName ? ` from ${context.fileName}` : '';
        const paramCount = countWeightsFromSpecs(weightSpecs);
        const formattedCount = Number.isFinite(paramCount) ? paramCount.toLocaleString() : 'unknown';
        log(`Loaded ${modelDisplayName(modelType)} model (${formattedCount} params)${origin}.`);
        if(wasRunning){
          log('Training paused. Restart AI training to continue with imported model.');
        }
        return;
      }

      const weights = snapshot.weights.map((v) => Number(v));
      let expectedDim = inputDimForModel(modelType);
      let snapshotHidden = [];
      if(isMlpModelType(modelType)){
        const raw = Array.isArray(snapshot.hiddenLayers) ? snapshot.hiddenLayers : [];
        snapshotHidden = raw.slice(0, MLP_MAX_HIDDEN_LAYERS).map((value, idx) => sanitizeHiddenUnits(value, idx, value));
        if(snapshotHidden.length === 0){
          const fallback = DEFAULT_MLP_HIDDEN[0] || 8;
          snapshotHidden.push(sanitizeHiddenUnits(fallback, 0, fallback));
        }
        let prev = inputDimForModel(modelType);
        expectedDim = 0;
        for(let i = 0; i < snapshotHidden.length; i++){
          const size = sanitizeHiddenUnits(snapshotHidden[i], i, snapshotHidden[i]);
          snapshotHidden[i] = size;
          expectedDim += prev * size + size;
          prev = size;
        }
        expectedDim += prev + 1;
      }

      if(weights.length !== expectedDim){
        throw new Error(`Expected ${expectedDim} weights for ${modelDisplayName(modelType)} model, received ${weights.length}`);
      }

      const masterWeights = allocFloat32(expectedDim);
      copyValues(weights, masterWeights);

      if(isMlpModelType(modelType)){
        applyMlpHiddenLayers(snapshotHidden, { rerenderControls: true, syncInputs: true });
      }

      train.modelType = modelType;
      currentModelType = modelType;
      train.dtype = dtype;
      dtypePreference = dtype;

      resetTraining();
      syncMlpConfigVisibility();

      train.mean = masterWeights;
      train.meanView = createDisplayView(train.mean, train.meanView);
      train.std = initialStd(modelType, mlpHiddenLayers);
      train.stdView = createDisplayView(train.std, train.stdView);
      train.currentWeightsOverride = null;
      resetAiPlanState();

      const bestCopy = new Float64Array(masterWeights.length);
      copyValues(masterWeights, bestCopy);
      train.bestEverWeights = bestCopy;
      train.bestEverFitness = Number.isFinite(snapshot.bestEverFitness) ? snapshot.bestEverFitness : -Infinity;
      const snapGen = Number(snapshot.generation);
      if(Number.isFinite(snapGen) && snapGen >= 0){
        train.gen = snapGen;
      }

      updateTrainStatus();

      const origin = context && context.fileName ? ` from ${context.fileName}` : '';
      log(`Loaded ${modelDisplayName(modelType)} weights (${masterWeights.length} params)${origin}.`);
      if(wasRunning){
        log('Training paused. Restart AI training to continue with imported weights.');
      }
    }

    function applyMlpHiddenLayers(newLayers, options = {}){
      const desiredLength = Math.max(
        MLP_MIN_HIDDEN_LAYERS,
        Math.min(MLP_MAX_HIDDEN_LAYERS, newLayers.length || 0),
      );
      const sanitized = [];
      for(let i = 0; i < desiredLength; i++){
        const fallback = mlpHiddenLayers[i];
        sanitized.push(sanitizeHiddenUnits(newLayers[i], i, fallback));
      }
      if(sanitized.length === 0){
        sanitized.push(sanitizeHiddenUnits(DEFAULT_MLP_HIDDEN[0], 0, DEFAULT_MLP_HIDDEN[0]));
      }
      const prevLayers = mlpHiddenLayers.slice();
      mlpHiddenLayers = sanitized;
      if(mlpHiddenCountSel){
        mlpHiddenCountSel.value = String(sanitized.length);
      }
      if(options.rerenderControls){
        renderMlpLayerControls();
      } else if(options.syncInputs !== false){
        for(let i = 0; i < sanitized.length; i++){
          const input = document.getElementById(`mlp-layer-${i}`);
          if(input && Number(input.value) !== sanitized[i]){
            input.value = sanitized[i];
          }
        }
      }
      const changed = sanitized.length !== prevLayers.length
        || sanitized.some((val, idx) => val !== prevLayers[idx]);
      if(train){
        train.mlpHiddenLayers = sanitized.slice();
        if(changed){
          if(isMlpModelType(train.modelType)){
            const wasRunning = train.enabled;
            resetTraining();
            if(wasRunning){
              startTraining();
            }
          } else {
            updateTrainStatus();
          }
        }
      }
      return sanitized;
    }

    function renderMlpLayerControls(){
      if(!mlpLayerControlsEl) return;
      mlpLayerControlsEl.innerHTML = '';
      for(let i = 0; i < mlpHiddenLayers.length; i++){
        const wrapper = document.createElement('div');
        wrapper.className = 'flex items-center gap-3';
        const inputId = `mlp-layer-${i}`;
        const label = document.createElement('label');
        label.setAttribute('for', inputId);
        label.className = 'text-xs uppercase tracking-[0.35em] text-plum/70';
        label.textContent = `Layer ${i + 1} Nodes`;
        const input = document.createElement('input');
        input.type = 'number';
        input.id = inputId;
        input.min = String(MLP_MIN_UNITS);
        input.max = String(MLP_MAX_UNITS);
        input.step = '1';
        input.value = mlpHiddenLayers[i];
        input.className = 'w-20 rounded-2xl border border-white/20 bg-white/10 px-3 py-1 text-center text-shell focus:outline-none focus:ring-2 focus:ring-citron/60';
        input.setAttribute('aria-label', `Hidden layer ${i + 1} nodes`);
        input.addEventListener('change', (e) => {
          const sanitized = sanitizeHiddenUnits(e.target.value, i, mlpHiddenLayers[i]);
          if(mlpHiddenLayers[i] === sanitized){
            e.target.value = sanitized;
            return;
          }
          const layers = mlpHiddenLayers.slice();
          layers[i] = sanitized;
          applyMlpHiddenLayers(layers, { syncInputs: false });
          e.target.value = sanitized;
        });
        wrapper.appendChild(label);
        wrapper.appendChild(input);
        mlpLayerControlsEl.appendChild(wrapper);
      }
    }

    function handleHiddenCountChange(value){
      const parsed = Math.round(Number(value));
      const desired = Number.isFinite(parsed)
        ? Math.max(MLP_MIN_HIDDEN_LAYERS, Math.min(MLP_MAX_HIDDEN_LAYERS, parsed))
        : mlpHiddenLayers.length;
      if(desired === mlpHiddenLayers.length){
        if(mlpHiddenCountSel){
          mlpHiddenCountSel.value = String(mlpHiddenLayers.length);
        }
        return;
      }
      const layers = mlpHiddenLayers.slice();
      if(desired > layers.length){
        const seed = layers.length
          ? layers[layers.length - 1]
          : sanitizeHiddenUnits(DEFAULT_MLP_HIDDEN[0], layers.length, DEFAULT_MLP_HIDDEN[0]);
        while(layers.length < desired){
          layers.push(seed);
        }
      } else {
        layers.length = desired;
      }
      applyMlpHiddenLayers(layers, { rerenderControls: true });
    }

    function syncMlpConfigVisibility(){
      if(!mlpConfigEl) return;
      if(train && isMlpModelType(train.modelType)){
        mlpConfigEl.classList.remove('hidden');
      } else {
        mlpConfigEl.classList.add('hidden');
      }
    }

    function initMlpConfigUi(){
      if(mlpHiddenCountSel){
        mlpHiddenCountSel.value = String(mlpHiddenLayers.length);
        mlpHiddenCountSel.addEventListener('change', (e) => {
          handleHiddenCountChange(e.target.value);
        });
      }
      renderMlpLayerControls();
      if(train){
        train.mlpHiddenLayers = mlpHiddenLayers.slice();
      }
      syncMlpConfigVisibility();
    }

    function renderAlphaNetworkPlaceholder(message){
      if(!networkVizEl){
        return;
      }
      networkVizEl.style.overflow = 'hidden';
      networkVizEl.style.overflowY = 'hidden';
      networkVizEl.style.overflowX = 'hidden';
      if(typeof document === 'undefined'){
        networkVizEl.textContent = message || `${ALPHATETRIS_MODEL_LABEL} visualization managed by TensorFlow.js.`;
        return;
      }
      networkVizEl.innerHTML = '';
      const container = document.createElement('div');
      container.className = 'alpha-network-placeholder';
      container.textContent = message || `${ALPHATETRIS_MODEL_LABEL} visualization managed by TensorFlow.js.`;
      networkVizEl.appendChild(container);
    }

    function showAlphaMetricMessage(message){
      if(alphaMetricPlotsEl){
        alphaMetricPlotsEl.innerHTML = '';
      }
      if(alphaMetricEmptyEl){
        if(typeof message === 'string' && message.trim()){
          alphaMetricEmptyEl.textContent = message.trim();
        }
        alphaMetricEmptyEl.classList.remove('is-hidden');
      }
    }

    function hideAlphaMetricMessage(){
      if(alphaMetricEmptyEl){
        alphaMetricEmptyEl.classList.add('is-hidden');
      }
    }

    function syncAlphaMetricSelect(selection){
      if(!alphaMetricSelectEl){
        return;
      }
      const desired = Array.isArray(selection)
        ? selection.filter((value, index, self) => self.indexOf(value) === index)
        : [];
      const options = Array.from(alphaMetricSelectEl.options || []);
      for(let i = 0; i < options.length; i += 1){
        const option = options[i];
        if(option){
          option.selected = desired.includes(option.value);
        }
      }
      if(alphaMetricChoices
        && typeof alphaMetricChoices.removeActiveItems === 'function'
        && typeof alphaMetricChoices.setChoiceByValue === 'function'){
        try {
          alphaMetricChoices.removeActiveItems();
          for(let i = 0; i < desired.length; i += 1){
            alphaMetricChoices.setChoiceByValue(desired[i]);
          }
        } catch (_) {
          /* ignore sync issues */
        }
      }
    }

    function appendAlphaMetricHistory(training, step, metrics){
      if(!training || !metrics || !Number.isFinite(step)){
        return false;
      }
      if(!training.metricsHistory || typeof training.metricsHistory !== 'object'){
        training.metricsHistory = {
          epoch: [],
          history: {},
          metrics: [],
          selected: alphaMetricSelection.slice(),
        };
      }
      const history = training.metricsHistory;
      const available = {};
      for(let i = 0; i < ALPHA_METRIC_NAMES.length; i += 1){
        const name = ALPHA_METRIC_NAMES[i];
        const value = metrics[name];
        if(Number.isFinite(value)){
          available[name] = value;
        }
      }
      const availableNames = Object.keys(available);
      if(availableNames.length === 0){
        return false;
      }
      if(!Array.isArray(history.metrics) || history.metrics.length === 0){
        history.metrics = availableNames.slice();
      } else {
        const filtered = history.metrics.filter((name) => availableNames.includes(name));
        const additions = availableNames.filter((name) => !filtered.includes(name));
        history.metrics = filtered.concat(additions);
      }
      let selectionChanged = false;
      if(!Array.isArray(history.selected) || history.selected.length === 0){
        const defaultSelected = alphaMetricSelection.length
          ? alphaMetricSelection.filter((name) => history.metrics.includes(name))
          : history.metrics.slice();
        history.selected = defaultSelected.length ? defaultSelected : history.metrics.slice();
        selectionChanged = true;
      } else {
        const filteredSelection = history.selected.filter((name) => history.metrics.includes(name));
        if(filteredSelection.length !== history.selected.length){
          history.selected = filteredSelection.length ? filteredSelection : history.metrics.slice();
          selectionChanged = true;
        }
      }
      alphaMetricSelection = history.selected.slice();
      if(selectionChanged){
        syncAlphaMetricSelect(alphaMetricSelection);
      }
      if(!Array.isArray(history.epoch)){
        history.epoch = [];
      }
      history.epoch.push(step);
      const tracked = history.metrics;
      for(let i = 0; i < tracked.length; i += 1){
        const name = tracked[i];
        if(!Array.isArray(history.history[name])){
          history.history[name] = [];
        }
        const value = available[name];
        history.history[name].push(value !== undefined ? value : null);
      }
      const maxPoints = Number.isFinite(training.metricHistoryLimit)
        ? Math.max(10, Math.floor(training.metricHistoryLimit))
        : ALPHA_METRIC_HISTORY_LIMIT;
      while(history.epoch.length > maxPoints){
        history.epoch.shift();
        for(let i = 0; i < tracked.length; i += 1){
          const name = tracked[i];
          const series = history.history[name];
          if(Array.isArray(series) && series.length){
            series.shift();
          }
        }
      }
      const trackedSet = new Set(tracked);
      Object.keys(history.history).forEach((name) => {
        if(!trackedSet.has(name)){
          delete history.history[name];
        }
      });
      return true;
    }

    function drawAlphaMetricSeries(canvas, steps, series, style){
      if(!canvas || !Array.isArray(series) || series.length === 0){
        return;
      }
      const ctx = canvas.getContext('2d');
      if(!ctx){
        return;
      }
      const ratio = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1;
      const rect = canvas.getBoundingClientRect();
      const baseWidth = rect && rect.width ? rect.width : canvas.width || 280;
      const baseHeight = rect && rect.height ? rect.height : canvas.height || 180;
      const width = Math.max(220, baseWidth);
      const height = Math.max(150, baseHeight);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, width, height);

      const background = ctx.createLinearGradient(0, 0, 0, height);
      background.addColorStop(0, 'rgba(94, 74, 227, 0.08)');
      background.addColorStop(1, 'rgba(12, 59, 46, 0.32)');
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, width, height);

      const margin = { top: 18, right: 24, bottom: 32, left: 56 };
      const chartWidth = Math.max(1, width - margin.left - margin.right);
      const chartHeight = Math.max(1, height - margin.top - margin.bottom);
      const finiteValues = series.filter((value) => Number.isFinite(value));
      if(!finiteValues.length){
        return;
      }
      let minY = Math.min(...finiteValues);
      let maxY = Math.max(...finiteValues);
      if(minY === maxY){
        const offset = Math.abs(minY) > 1e-6 ? Math.abs(minY) * 0.2 : 1;
        minY -= offset;
        maxY += offset;
      } else {
        const pad = (maxY - minY) * 0.12;
        minY -= pad;
        maxY += pad;
      }
      if(!Number.isFinite(minY) || !Number.isFinite(maxY)){
        return;
      }
      const firstStep = Array.isArray(steps) && steps.length && Number.isFinite(steps[0]) ? steps[0] : 0;
      const lastStep = Array.isArray(steps) && steps.length && Number.isFinite(steps[steps.length - 1])
        ? steps[steps.length - 1]
        : firstStep + series.length - 1;
      const stepRange = lastStep - firstStep;
      const fallbackDenom = series.length > 1 ? series.length - 1 : 1;

      const computeX = (index) => {
        const step = Array.isArray(steps) && steps.length > index && Number.isFinite(steps[index])
          ? steps[index]
          : firstStep + index;
        const norm = stepRange > 0 ? (step - firstStep) / stepRange : (fallbackDenom ? index / fallbackDenom : 0);
        return margin.left + norm * chartWidth;
      };
      const computeY = (value) => {
        const denom = maxY - minY || 1;
        const norm = (value - minY) / denom;
        return margin.top + (1 - norm) * chartHeight;
      };

      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(249, 245, 255, 0.12)';
      ctx.setLineDash([4, 6]);
      const gridLines = 4;
      for(let i = 0; i <= gridLines; i += 1){
        const y = margin.top + (chartHeight * i) / gridLines;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + chartWidth, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      if(minY < 0 && maxY > 0){
        const zeroNorm = (0 - minY) / (maxY - minY || 1);
        const zeroY = margin.top + chartHeight - zeroNorm * chartHeight;
        ctx.strokeStyle = 'rgba(249, 245, 255, 0.32)';
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(margin.left, zeroY);
        ctx.lineTo(margin.left + chartWidth, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const stroke = style && style.stroke ? style.stroke : 'rgba(244, 247, 121, 0.92)';
      const fillColor = style && style.fill ? style.fill : 'rgba(244, 247, 121, 0.18)';
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = stroke;
      const segments = [];
      let segment = [];
      let lastPoint = null;
      for(let i = 0; i < series.length; i += 1){
        const value = series[i];
        if(!Number.isFinite(value)){
          if(segment.length){
            segments.push(segment);
            segment = [];
          }
          continue;
        }
        const x = computeX(i);
        const y = computeY(value);
        segment.push({ x, y });
        lastPoint = { x, y };
      }
      if(segment.length){
        segments.push(segment);
      }
      const gradientFill = ctx.createLinearGradient(0, margin.top, 0, margin.top + chartHeight);
      gradientFill.addColorStop(0, fillColor);
      gradientFill.addColorStop(1, 'rgba(15, 19, 35, 0)');
      for(let s = 0; s < segments.length; s += 1){
        const points = segments[s];
        if(points.length === 0){
          continue;
        }
        if(points.length === 1){
          ctx.beginPath();
          ctx.fillStyle = stroke;
          ctx.arc(points[0].x, points[0].y, 4, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for(let j = 1; j < points.length; j += 1){
          ctx.lineTo(points[j].x, points[j].y);
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(points[0].x, margin.top + chartHeight);
        for(let j = 0; j < points.length; j += 1){
          ctx.lineTo(points[j].x, points[j].y);
        }
        ctx.lineTo(points[points.length - 1].x, margin.top + chartHeight);
        ctx.closePath();
        ctx.fillStyle = gradientFill;
        ctx.fill();
      }
      if(lastPoint){
        ctx.save();
        ctx.fillStyle = stroke;
        ctx.beginPath();
        ctx.arc(lastPoint.x, lastPoint.y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = 'rgba(15, 19, 35, 0.82)';
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = 'rgba(249, 245, 255, 0.6)';
      ctx.font = '12px "Instrument Serif", serif';
      ctx.textAlign = 'right';
      ctx.fillText(maxY.toFixed(3), margin.left - 8, margin.top + 4);
      ctx.fillText(minY.toFixed(3), margin.left - 8, margin.top + chartHeight);
      ctx.textAlign = 'center';
      ctx.fillText(String(firstStep), margin.left, margin.top + chartHeight + 22);
      ctx.fillText(String(lastStep), margin.left + chartWidth, margin.top + chartHeight + 22);
    }

    function renderAlphaMetricHistory(training){
      if(!alphaMetricPlotsEl){
        return;
      }
      if(!training || !training.metricsHistory){
        showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');
        return;
      }
      const history = training.metricsHistory;
      const steps = Array.isArray(history.epoch) ? history.epoch.slice() : [];
      const tracked = Array.isArray(history.metrics) ? history.metrics.slice() : [];
      if(!steps.length || !tracked.length){
        showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');
        return;
      }
      let selected = Array.isArray(history.selected) && history.selected.length
        ? history.selected.filter((name) => tracked.includes(name))
        : alphaMetricSelection.filter((name) => tracked.includes(name));
      if(!selected.length){
        showAlphaMetricMessage('Select at least one metric to plot.');
        return;
      }
      history.selected = selected.slice();
      alphaMetricSelection = history.selected.slice();
      hideAlphaMetricMessage();
      alphaMetricPlotsEl.innerHTML = '';
      let rendered = 0;
      for(let i = 0; i < selected.length; i += 1){
        const name = selected[i];
        const series = history.history[name];
        if(!Array.isArray(series) || series.length !== steps.length){
          continue;
        }
        const card = document.createElement('div');
        card.className = 'alpha-metric-card';

        const header = document.createElement('div');
        header.className = 'alpha-metric-card__header';

        const title = document.createElement('div');
        title.className = 'alpha-metric-card__title';
        const swatch = document.createElement('span');
        swatch.className = 'alpha-metric-card__swatch';
        const style = ALPHA_METRIC_STYLES[name] || null;
        if(style && style.stroke){
          swatch.style.setProperty('--alpha-metric-color', style.stroke);
        }
        title.appendChild(swatch);
        const nameEl = document.createElement('span');
        nameEl.className = 'alpha-metric-card__name';
        nameEl.textContent = ALPHA_METRIC_LABELS[name] || name;
        title.appendChild(nameEl);
        header.appendChild(title);

        const stats = document.createElement('div');
        stats.className = 'alpha-metric-card__stats';
        const latest = series[series.length - 1];
        const valueEl = document.createElement('div');
        valueEl.className = 'alpha-metric-card__value';
        valueEl.textContent = Number.isFinite(latest) ? latest.toFixed(4) : '—';
        stats.appendChild(valueEl);
        const deltaEl = document.createElement('div');
        deltaEl.className = 'alpha-metric-card__delta';
        const first = series[0];
        if(Number.isFinite(first) && Number.isFinite(latest)){
          const diff = latest - first;
          const prefix = diff > 0 ? '▲' : diff < 0 ? '▼' : '±';
          deltaEl.textContent = `${prefix}${Math.abs(diff).toFixed(4)}`;
          if(diff > 0){
            deltaEl.dataset.trend = 'up';
          } else if(diff < 0){
            deltaEl.dataset.trend = 'down';
          } else {
            deltaEl.dataset.trend = 'flat';
          }
        } else {
          deltaEl.textContent = '—';
          deltaEl.dataset.trend = 'flat';
        }
        stats.appendChild(deltaEl);
        header.appendChild(stats);
        card.appendChild(header);

        const meta = document.createElement('div');
        meta.className = 'alpha-metric-card__meta';
        const lastStep = steps[steps.length - 1];
        meta.textContent = Number.isFinite(lastStep) ? `Step ${lastStep}` : '';
        card.appendChild(meta);

        const canvas = document.createElement('canvas');
        canvas.className = 'alpha-metric-chart';
        card.appendChild(canvas);

        alphaMetricPlotsEl.appendChild(card);
        drawAlphaMetricSeries(canvas, steps, series, style);
        rendered += 1;
      }
      if(rendered === 0){
        showAlphaMetricMessage('ConvNet losses will appear once AlphaTetris training runs.');
      }
    }

    function captureAlphaConvSummary(model){
      const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
      if(!model || !tf || !Array.isArray(model.layers)){
        return null;
      }
      const layers = [];
      for(let i = 0; i < model.layers.length; i += 1){
        const layer = model.layers[i];
        if(!layer || typeof layer.getClassName !== 'function'){
          continue;
        }
        if(layer.getClassName() !== 'Conv2D'){
          continue;
        }
        let weights;
        try {
          weights = layer.getWeights();
        } catch (_) {
          weights = null;
        }
        if(!weights || !weights.length){
          continue;
        }
        const kernel = weights[0];
        const shape = kernel && Array.isArray(kernel.shape) ? kernel.shape : [];
        if(shape.length !== 4){
          weights.forEach((tensor) => { if(tensor && typeof tensor.dispose === 'function'){ tensor.dispose(); } });
          continue;
        }
        const [kernelHeight, kernelWidth, inChannels, outChannels] = shape.map((dim) => Math.max(1, Math.floor(dim)));
        let data;
        try {
          data = Float32Array.from(kernel.dataSync());
        } catch (err) {
          if(typeof console !== 'undefined' && typeof console.warn === 'function'){
            console.warn('Failed to read convolution weights', err);
          }
          weights.forEach((tensor) => { if(tensor && typeof tensor.dispose === 'function'){ tensor.dispose(); } });
          continue;
        }
        weights.forEach((tensor) => { if(tensor && typeof tensor.dispose === 'function'){ tensor.dispose(); } });
        const filterCount = Math.max(1, outChannels);
        const filters = [];
        let globalMin = Infinity;
        let globalMax = -Infinity;
        for(let filter = 0; filter < filterCount; filter += 1){
          const values = new Float32Array(kernelHeight * kernelWidth);
          let min = Infinity;
          let max = -Infinity;
          for(let y = 0; y < kernelHeight; y += 1){
            for(let x = 0; x < kernelWidth; x += 1){
              let sum = 0;
              for(let c = 0; c < inChannels; c += 1){
                const idx = (((y * kernelWidth) + x) * inChannels + c) * filterCount + filter;
                sum += data[idx] || 0;
              }
              const avg = sum / Math.max(1, inChannels);
              const offset = y * kernelWidth + x;
              values[offset] = avg;
              if(avg < min) min = avg;
              if(avg > max) max = avg;
            }
          }
          globalMin = Math.min(globalMin, min);
          globalMax = Math.max(globalMax, max);
          filters.push({
            index: filter,
            min,
            max,
            maxAbs: Math.max(Math.abs(min), Math.abs(max)),
            values,
          });
        }
        layers.push({
          name: typeof layer.name === 'string' ? layer.name : `Conv2D ${layers.length + 1}`,
          kernelSize: [kernelHeight, kernelWidth],
          inChannels,
          outChannels: filterCount,
          min: Number.isFinite(globalMin) ? globalMin : null,
          max: Number.isFinite(globalMax) ? globalMax : null,
          maxAbs: Math.max(Math.abs(globalMin), Math.abs(globalMax)),
          filters,
        });
      }
      if(!layers.length){
        return null;
      }
      return { step: null, layers };
    }

    function lerpColor(from, to, t){
      const clamped = Math.max(0, Math.min(1, t));
      return [
        Math.round(from[0] + (to[0] - from[0]) * clamped),
        Math.round(from[1] + (to[1] - from[1]) * clamped),
        Math.round(from[2] + (to[2] - from[2]) * clamped),
      ];
    }

    function mapAlphaConvColor(norm){
      if(!Number.isFinite(norm)){
        return [229, 231, 235];
      }
      const neutral = [229, 231, 235];
      const positive = [249, 115, 22];
      const negative = [59, 130, 246];
      const clamped = Math.max(-1, Math.min(1, norm));
      if(clamped >= 0){
        return lerpColor(neutral, positive, clamped);
      }
      return lerpColor(neutral, negative, -clamped);
    }

    function drawAlphaConvFilter(canvas, values, kernelWidth, kernelHeight, scale){
      if(!canvas || !values){
        return;
      }
      const ctx = canvas.getContext('2d');
      if(!ctx){
        return;
      }
      const width = Math.max(1, kernelWidth);
      const height = Math.max(1, kernelHeight);
      const imageData = ctx.createImageData(width, height);
      const denom = Number.isFinite(scale) && scale > 0 ? scale : 1;
      for(let y = 0; y < height; y += 1){
        for(let x = 0; x < width; x += 1){
          const value = values[y * width + x] || 0;
          const norm = Math.max(-1, Math.min(1, value / denom));
          const [r, g, b] = mapAlphaConvColor(norm);
          const idx = (y * width + x) * 4;
          imageData.data[idx] = r;
          imageData.data[idx + 1] = g;
          imageData.data[idx + 2] = b;
          imageData.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }

    function renderAlphaConvFilters(summary, options = {}){
      if(!networkVizEl){
        return;
      }
      if(!summary || !Array.isArray(summary.layers) || !summary.layers.length){
        renderAlphaNetworkPlaceholder('ConvNet filters will appear after a few training updates.');
        return;
      }
      networkVizEl.innerHTML = '';
      networkVizEl.style.overflowY = 'auto';
      networkVizEl.style.overflowX = 'hidden';

      const container = document.createElement('div');
      container.className = 'alpha-conv-viz';

      const stepSource = Number.isFinite(options.step) ? options.step : (Number.isFinite(summary.step) ? summary.step : null);
      const header = document.createElement('div');
      header.className = 'alpha-conv-viz__meta';
      const stepLabel = stepSource ? `Step ${stepSource}` : 'Latest weights';
      const metrics = options.metrics || {};
      const metricParts = [];
      if(Number.isFinite(metrics.loss)) metricParts.push(`Loss ${metrics.loss.toFixed(4)}`);
      if(Number.isFinite(metrics.policy_loss)) metricParts.push(`Policy ${metrics.policy_loss.toFixed(4)}`);
      if(Number.isFinite(metrics.value_loss)) metricParts.push(`Value ${metrics.value_loss.toFixed(4)}`);
      header.textContent = metricParts.length ? `${stepLabel} • ${metricParts.join(' • ')}` : stepLabel;
      container.appendChild(header);

      for(let i = 0; i < summary.layers.length; i += 1){
        const layer = summary.layers[i];
        const layerBox = document.createElement('div');
        layerBox.className = 'alpha-conv-layer';

        const title = document.createElement('div');
        title.className = 'alpha-conv-layer__title';
        const [kh, kw] = Array.isArray(layer.kernelSize) ? layer.kernelSize : [null, null];
        const kernelLabel = Number.isFinite(kh) && Number.isFinite(kw) ? `${kh}×${kw}` : 'kernel';
        title.textContent = `${layer.name || `Conv ${i + 1}`} — ${kernelLabel} • ${layer.outChannels || (layer.filters ? layer.filters.length : '?')} filters`;
        layerBox.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'alpha-conv-layer__grid';
        layerBox.appendChild(grid);

        const filters = Array.isArray(layer.filters) ? layer.filters : [];
        const maxAbs = Number.isFinite(layer.maxAbs) && layer.maxAbs > 0
          ? layer.maxAbs
          : filters.reduce((acc, filter) => {
              const candidate = Number.isFinite(filter.maxAbs) ? filter.maxAbs : 0;
              return candidate > acc ? candidate : acc;
            }, 0) || 1;

        for(let j = 0; j < filters.length; j += 1){
          const filter = filters[j];
          const filterBox = document.createElement('div');
          filterBox.className = 'alpha-conv-filter';
          const canvas = document.createElement('canvas');
          canvas.className = 'alpha-conv-filter__canvas';
          const kernelHeight = Array.isArray(layer.kernelSize) ? layer.kernelSize[0] : null;
          const kernelWidth = Array.isArray(layer.kernelSize) ? layer.kernelSize[1] : null;
          const width = Number.isFinite(kernelWidth) ? kernelWidth : Math.max(1, Math.round(Math.sqrt(filter.values ? filter.values.length : 1)));
          const height = Number.isFinite(kernelHeight) ? kernelHeight : width;
          canvas.width = Math.max(1, width);
          canvas.height = Math.max(1, height);
          drawAlphaConvFilter(canvas, filter.values, canvas.width, canvas.height, filter.maxAbs || maxAbs || 1);
          filterBox.appendChild(canvas);
          const label = document.createElement('div');
          label.className = 'alpha-conv-filter__label';
          label.textContent = `#${(Number.isFinite(filter.index) ? filter.index : j) + 1}`;
          filterBox.appendChild(label);
          grid.appendChild(filterBox);
        }

        container.appendChild(layerBox);
      }

      networkVizEl.appendChild(container);
    }

    function recordAlphaMilestoneSnapshot(alphaState, training, model, milestoneStep, metrics){
      if(!train || !isAlphaModelType(train.modelType)){
        return false;
      }
      const summary = captureAlphaConvSummary(model);
      if(!summary){
        return false;
      }
      if(Number.isFinite(milestoneStep)){
        summary.step = milestoneStep;
      }
      training.lastConvSummary = summary;
      training.lastConvSummaryStep = summary.step;
      training.lastConvMetrics = metrics || null;
      const alphaConfig = alphaState && alphaState.config ? alphaState.config : {};
      const rawDescription = typeof alphaConfig.architectureDescription === 'string'
        ? alphaConfig.architectureDescription
        : ALPHATETRIS_DEFAULT_ARCHITECTURE;
      const architectureDescription = normalizeAlphaArchitectureDescription(rawDescription)
        || ALPHATETRIS_DEFAULT_ARCHITECTURE;
      const snapshot = {
        modelType: 'alphatetris',
        dtype: 'f32',
        step: summary.step,
        gen: summary.step,
        metrics: metrics || null,
        convSummary: summary,
        architectureDescription,
      };
      recordGenerationSnapshot(snapshot);
      if(Array.isArray(train.bestByGeneration) && train.bestByGeneration.length > MAX_ALPHA_SNAPSHOTS){
        const excess = train.bestByGeneration.length - MAX_ALPHA_SNAPSHOTS;
        if(excess > 0){
          train.bestByGeneration.splice(0, excess);
          if(train.historySelection !== null){
            train.historySelection = Math.max(0, train.historySelection - excess);
          }
          syncHistoryControls();
        }
      }
      return true;
    }

    function sliceSegment(arr, start, end){
      if(!arr) return null;
      if(typeof arr.subarray === 'function'){
        return arr.subarray(start, end);
      }
      return arr.slice(start, end);
    }

    function inferLayerSizesFromWeights(weights, override, inputDimOverride){
      const inputDim = Number.isFinite(inputDimOverride) ? inputDimOverride : FEATURE_NAMES.length;
      if(Array.isArray(override) && override.length >= 2){
        return override.slice();
      }
      if(!weights || !weights.length){
        return [inputDim, 1];
      }
      const total = weights.length;
      const cache = new Map();

      function dfs(offset, prev){
        if(offset === total){
          return [];
        }
        const key = `${offset}|${prev}`;
        if(cache.has(key)) return cache.get(key);
        const remaining = total - offset;

        if(remaining % prev === 0){
          const outSize = remaining / prev;
          const seq = [outSize];
          cache.set(key, seq);
          return seq;
        }

        const maxNext = Math.floor(remaining / (prev + 1));
        for(let next = maxNext; next >= 1; next--){
          const need = prev * next + next;
          if(need > remaining) continue;
          const rest = dfs(offset + need, next);
          if(rest){
            const seq = [next, ...rest];
            cache.set(key, seq);
            return seq;
          }
        }
        cache.set(key, null);
        return null;
      }

      const seq = dfs(0, inputDim);
      if(seq){
        return [inputDim, ...seq];
      }
      if(total === inputDim){
        return [inputDim, 1];
      }
      if(total === inputDim + 1){
        return [inputDim, 1];
      }
      return [inputDim, 1];
    }

    function sliceWeightMatrices(weights, layerSizes){
      const slices = [];
      if(!weights || !layerSizes || layerSizes.length < 2) return slices;
      let offset = 0;
      const totalLen = weights.length || 0;
      for(let layer = 1; layer < layerSizes.length; layer++){
        const prev = layerSizes[layer - 1];
        const curr = layerSizes[layer];
        const weightCount = prev * curr;
        const matrix = sliceSegment(weights, offset, offset + weightCount);
        offset += weightCount;
        let bias = null;
        if(offset + curr <= totalLen){
          bias = sliceSegment(weights, offset, offset + curr);
          offset += curr;
        }
        slices.push({ weights: matrix, bias });
      }
      return slices;
    }

    function renderNetworkD3(weights, overrideLayerSizes, options = {}){
      if(!networkVizEl || typeof d3 === 'undefined'){
        return;
      }
      networkVizEl.style.overflow = 'hidden';
      networkVizEl.style.overflowY = 'hidden';
      networkVizEl.style.overflowX = 'hidden';
      const width = networkVizEl.clientWidth || 320;
      const height = networkVizEl.clientHeight || 220;
      const marginX = 52;
      const marginY = 28;

      let svg = d3.select(networkVizEl).select('svg');
      if(svg.empty()){
        svg = d3.select(networkVizEl)
          .append('svg')
          .attr('role', 'img')
          .attr('aria-label', 'Visualization of model weights')
          .attr('preserveAspectRatio', 'xMidYMid meet');
      }
      svg
        .attr('width', width)
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);
      svg.selectAll('*').remove();

      if(!weights || !weights.length){
        svg.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('fill', '#888')
          .text('Weights unavailable');
        return;
      }

      const layerSizes = inferLayerSizesFromWeights(weights, overrideLayerSizes, options.inputDim);
      const slices = sliceWeightMatrices(weights, layerSizes);
      const totalLayers = layerSizes.length;
      const innerWidth = Math.max(width - 2 * marginX, 10);
      const innerHeight = Math.max(height - 2 * marginY, 10);

      const nodes = [];
      const nodeLookup = new Map();
      const featureNames = Array.isArray(options.featureNames) ? options.featureNames : FEATURE_NAMES;
      for(let layerIdx = 0; layerIdx < totalLayers; layerIdx++){
        const layerSize = layerSizes[layerIdx];
        const x = totalLayers === 1 ? width / 2 : marginX + (innerWidth * layerIdx) / Math.max(1, totalLayers - 1);
        const step = layerSize > 1 ? innerHeight / (layerSize - 1) : 0;
        const biasSlice = layerIdx > 0 && slices[layerIdx - 1] ? slices[layerIdx - 1].bias : null;
        for(let i = 0; i < layerSize; i++){
          const y = layerSize > 1 ? marginY + step * i : height / 2;
          const id = `${layerIdx}-${i}`;
          const label = layerIdx === 0
            ? (featureNames[i] || `x${i + 1}`)
            : (layerIdx === totalLayers - 1
              ? (layerSize === 1 ? 'Output' : `y${i + 1}`)
              : `h${layerIdx}-${i + 1}`);
          const biasVal = (biasSlice && biasSlice.length > i) ? biasSlice[i] : null;
          const node = { id, layer: layerIdx, index: i, x, y, label, bias: biasVal };
          nodes.push(node);
          nodeLookup.set(id, node);
        }
      }

      const edges = [];
      for(let layerIdx = 1; layerIdx < layerSizes.length; layerIdx++){
        const prev = layerSizes[layerIdx - 1];
        const curr = layerSizes[layerIdx];
        const slice = slices[layerIdx - 1];
        const matrix = slice && slice.weights ? slice.weights : null;
        for(let i = 0; i < prev; i++){
          for(let j = 0; j < curr; j++){
            const wIdx = matrix ? (i * curr + j) : null;
            const weightValue = (matrix && wIdx !== null && wIdx < matrix.length) ? matrix[wIdx] : 0;
            edges.push({
              source: nodeLookup.get(`${layerIdx - 1}-${i}`),
              target: nodeLookup.get(`${layerIdx}-${j}`),
              weight: weightValue,
            });
          }
        }
      }

      const maxAbs = edges.length ? d3.max(edges, (d) => Math.abs(d.weight)) : 0;
      const denom = (maxAbs && Number.isFinite(maxAbs) && maxAbs > 0) ? maxAbs : 1;

      const edgeGroup = svg.append('g').attr('class', 'edges');
      const edgeSel = edgeGroup.selectAll('line')
        .data(edges)
        .enter()
        .append('line')
        .attr('x1', (d) => d.source ? d.source.x : 0)
        .attr('y1', (d) => d.source ? d.source.y : 0)
        .attr('x2', (d) => d.target ? d.target.x : 0)
        .attr('y2', (d) => d.target ? d.target.y : 0)
        .attr('stroke', (d) => (d.weight >= 0 ? '#2b8cbe' : '#d7301f'))
        .attr('stroke-width', (d) => {
          const norm = Math.min(1, Math.abs(d.weight) / denom);
          return 0.6 + norm * 3.4;
        })
        .attr('stroke-opacity', (d) => {
          const norm = Math.min(1, Math.abs(d.weight) / denom);
          return 0.2 + norm * 0.8;
        });
      edgeSel.append('title').text((d) => `w=${d.weight.toFixed(3)}`);

      const nodeGroup = svg.append('g').attr('class', 'nodes');
      const nodeSel = nodeGroup.selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('transform', (d) => `translate(${d.x}, ${d.y})`);

      nodeSel.append('circle')
        .attr('r', 10)
        .attr('fill', '#fff')
        .attr('stroke', '#555')
        .attr('stroke-width', 1.2);

      nodeSel.append('text')
        .attr('text-anchor', (d) => {
          if(d.layer === 0) return 'end';
          if(d.layer === totalLayers - 1) return 'start';
          return 'middle';
        })
        .attr('x', (d) => {
          if(d.layer === 0) return -14;
          if(d.layer === totalLayers - 1) return 14;
          return 0;
        })
        .attr('dy', 4)
        .text((d) => d.label);

      nodeSel.append('title').text((d) => {
        if(d.layer === 0){
          return d.label;
        }
        if(typeof d.bias === 'number' && Number.isFinite(d.bias)){
          return `${d.label} (bias ${d.bias.toFixed(3)})`;
        }
        return d.label;
      });
    }

    const train = {
      enabled: false,
      gen: 0,
      popSize: 16,
      eliteFrac: 0.25,
      modelType: 'linear',
      mlpHiddenLayers: mlpHiddenLayers.slice(),
      dtype: DEFAULT_DTYPE,
      mean: initialMean('linear'),
      std: initialStd('linear'),
      minStd: 0.05,
      maxStd: 3.0,
      candWeights: [],
      candWeightViews: [],
      candScores: [],
      candIndex: -1,
      // Visualization + speed controls for training
      visualizeBoard: false,     // if false: skip board/preview rendering
      plotBestOnly: true,
      currentWeightsOverride: null,
      ai: {
        plan: null,
        acc: 0,
        lastSig: '',
        staleMs: 0,
        lastSearchStats: null,
        search: {
          simulations: 48,
          cPuct: 1.5,
          temperature: 1.0,
          discount: 1,
        },
      },
      performanceSummary: [],
      gameScores: [],
      gameModelTypes: [],
      gameScoresOffset: 0,
      totalGamesPlayed: 0,
      bestEverFitness: -Infinity,
      bestEverWeights: null,
      bestByGeneration: [],
      historySelection: null,
      maxPlotPoints: 4000,
      scorePlotUpdateFreq: SCORE_PLOT_DEFAULT_UPDATE_FREQ,
      scorePlotPending: 0,
      scorePlotAxisMax: 0,
      meanView: null,
      stdView: null,
      alpha: null,
    };
    function shouldLogTrainingEvent(){
      return !(train && train.enabled && train.visualizeBoard === false);
    }
    function logTrainingEvent(message){
      if(!shouldLogTrainingEvent()){
        return;
      }
      log(message);
    }
    const gridScratch = Array.from({ length: HEIGHT }, () => Array(WIDTH).fill(0));
    const columnHeightScratch = new Array(WIDTH).fill(0);
    const columnMaskScratch = typeof Uint32Array !== 'undefined' ? new Uint32Array(WIDTH) : new Array(WIDTH).fill(0);
    const metricsScratch = {
      holes: 0,
      bump: 0,
      maxHeight: 0,
      wellSum: 0,
      edgeWell: 0,
      tetrisWell: 0,
      contact: 0,
      rowTransitions: 0,
      colTransitions: 0,
      aggregateHeight: 0,
    };
    const baselineColumnMaskScratch =
      typeof Uint32Array !== 'undefined' ? new Uint32Array(WIDTH) : new Array(WIDTH).fill(0);
    const baselineColumnHeightScratch = new Array(WIDTH).fill(0);
    const baselineRowMaskScratch =
      typeof Uint16Array !== 'undefined' ? new Uint16Array(HEIGHT) : new Array(HEIGHT).fill(0);
    const surrogateRowMaskScratch =
      typeof Uint16Array !== 'undefined' ? new Uint16Array(HEIGHT) : new Array(HEIGHT).fill(0);
    const surrogateRowMaskCollapsedScratch =
      typeof Uint16Array !== 'undefined' ? new Uint16Array(HEIGHT) : new Array(HEIGHT).fill(0);
    const surrogateColumnMaskEstimateScratch =
      typeof Uint32Array !== 'undefined' ? new Uint32Array(WIDTH) : new Array(WIDTH).fill(0);
    const surrogateColumnHeightEstimateScratch =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(WIDTH) : new Array(WIDTH).fill(0);
    const surrogateRowTouchedFlags =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(HEIGHT) : new Array(HEIGHT).fill(0);
    const surrogateClearedRowFlags =
      typeof Uint8Array !== 'undefined' ? new Uint8Array(HEIGHT) : new Array(HEIGHT).fill(0);
    const surrogateMetricsScratch = {
      holes: 0,
      bump: 0,
      maxHeight: 0,
      wellSum: 0,
      edgeWell: 0,
      tetrisWell: 0,
      contact: 0,
      rowTransitions: 0,
      colTransitions: 0,
      aggregateHeight: 0,
    };
    const surrogateClearedRowsScratch = [];
    const placementSurrogateKeyCache = new Map();
    const placementSurrogate = {
      enabled: true,
      debugCompare: false,
      stats: {
        attempts: 0,
        successes: 0,
        fallbacks: 0,
        mismatches: 0,
        comparisons: 0,
        timeSurrogate: 0,
        timeFallback: 0,
        timeDebugCompare: 0,
      },
      nextLogThreshold: 500,
    };
    const surrogateTimerNow =
      typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? () => performance.now()
        : () => Date.now();
    train.placementSurrogate = placementSurrogate;
    const pieceBottomProfiles = Object.fromEntries(
      Object.entries(SHAPES).map(([shape, rotations]) => {
        const perRotation = rotations.map((state) => {
          const columnBottoms = new Map();
          for (let i = 0; i < state.length; i += 1) {
            const [rowOffset, colOffset] = state[i];
            const prev = columnBottoms.get(colOffset);
            if (prev === undefined || rowOffset > prev) {
              columnBottoms.set(colOffset, rowOffset);
            }
          }
          const entries = [];
          for (const [colIndex, bottomRow] of columnBottoms.entries()) {
            entries.push({ col: colIndex, bottom: bottomRow });
          }
          entries.sort((a, b) => a.col - b.col);
          return entries;
        });
        return [shape, perRotation];
      })
    );
    const clearedRowsScratch = [];
    const featureScratch = new Float32Array(FEAT_DIM);
    const rawFeatureScratch = new Float32Array(RAW_FEAT_DIM);
    const pooledPiece = new Piece('I');
    const alphaSpawnPiece = new Piece('I');
    const simulateResultScratch = { lines: 0, grid: gridScratch, dropRow: 0, clearedRows: clearedRowsScratch, clearedRowCount: 0 };
    window.__train = train;
    window.__placementSurrogate = placementSurrogate;
    // After `train` exists, honor train.dtype for future allocations
    dtypePreference = train.dtype || DEFAULT_DTYPE;
    train.meanView = createDisplayView(train.mean, train.meanView);
    train.stdView = createDisplayView(train.std, train.stdView);
    train.candWeightViews = [];

    const initialAxisSeed = usesPopulationModel(train.modelType)
      ? Math.max(1, Number.isFinite(train.popSize) ? train.popSize : 0)
      : 10;
    train.scorePlotAxisMax = Math.max(10, Math.ceil(initialAxisSeed * 1.2));
    train.scorePlotPending = 0;
    syncHistoryControls();
    syncMctsControls();

    function getDisplayWeightsForUi(weights, options = {}){
      if(!weights) return weights;
      const dtype = options.dtype || dtypePreference;
      if(dtype !== 'f16' || !HAS_F16){
        return weights;
      }
      if(weights === train.mean && train.meanView){
        return train.meanView;
      }
      if(weights === train.std && train.stdView){
        return train.stdView;
      }
      if(Array.isArray(train.candWeights) && Array.isArray(train.candWeightViews)){
        const idx = train.candWeights.indexOf(weights);
        if(idx >= 0 && train.candWeightViews[idx]){
          return train.candWeightViews[idx];
        }
      }
      const length = (typeof weights.length === 'number') ? weights.length : 0;
      if(length === 0){
        return weights;
      }
      const view = new Float16Array(length);
      copyValues(weights, view);
      return view;
    }

    function updateTrainStatus(){
      if(trainStatus){
        const statusLabel = modelDisplayName(train.modelType);
        const populationModel = usesPopulationModel(train.modelType);
        if(train.enabled){
          if(populationModel){
            const maxIndex = Math.max(0, (Array.isArray(train.candWeights) ? train.candWeights.length : 0) - 1);
            const safeIndex = Number.isFinite(train.candIndex) ? Math.max(0, Math.min(maxIndex, Math.floor(train.candIndex))) : 0;
            const candidateNumber = safeIndex + 1;
            const popSize = Number.isFinite(train.popSize) && train.popSize > 0 ? train.popSize : (maxIndex + 1 || 1);
            trainStatus.textContent = `Gen ${train.gen+1}, Candidate ${candidateNumber}/${popSize} — Model: ${statusLabel}`;
          } else {
            const gamesPlayed = Number.isFinite(train.totalGamesPlayed) ? train.totalGamesPlayed : 0;
            const nextGame = gamesPlayed + 1;
            trainStatus.textContent = `Training active — Model: ${statusLabel} (Game ${nextGame})`;
          }
        } else {
          trainStatus.textContent = `Training stopped — Model: ${statusLabel}`;
        }
      }
      const historySelection = getHistorySelection();
      const snapshot = historySelection ? historySelection.entry : null;
      let currentWeights = null;
      let overrideLayers = null;

      if(snapshot){
        currentWeights = snapshot.weights || null;
        if(Array.isArray(snapshot.layerSizes) && snapshot.layerSizes.length >= 2){
          overrideLayers = snapshot.layerSizes.slice();
        } else if(isMlpModelType(snapshot.modelType)){
          overrideLayers = currentMlpLayerSizes(snapshot.modelType);
        } else if(snapshot.modelType === 'linear'){
          overrideLayers = [FEAT_DIM, 1];
        }
      } else if(train.currentWeightsOverride){
        currentWeights = train.currentWeightsOverride;
        if(isMlpModelType(train.modelType)){
          overrideLayers = currentMlpLayerSizes(train.modelType);
        }
      } else if(usesPopulationModel(train.modelType) && train.enabled && train.candIndex >= 0 && train.candIndex < train.candWeights.length){
        currentWeights = train.candWeights[train.candIndex];
        if(isMlpModelType(train.modelType)){
          overrideLayers = currentMlpLayerSizes(train.modelType);
        }
      } else if(usesPopulationModel(train.modelType) && train.mean){
        currentWeights = train.mean;
        if(isMlpModelType(train.modelType)){
          overrideLayers = currentMlpLayerSizes(train.modelType);
        }
      }

      if(!snapshot && !currentWeights && usesPopulationModel(train.modelType) && train.bestEverWeights){
        currentWeights = train.bestEverWeights;
        overrideLayers = isMlpModelType(train.modelType) ? currentMlpLayerSizes(train.modelType) : [FEAT_DIM, 1];
      }

      const fallbackModelType = train ? train.modelType : currentModelType;
      const displayModelType = snapshot && snapshot.modelType ? snapshot.modelType : fallbackModelType;
      const displayUsesPopulation = usesPopulationModel(displayModelType);

      if(architectureEl){
        if(snapshot){
          architectureEl.textContent = describeSnapshotArchitecture(snapshot);
        } else {
          architectureEl.textContent = describeModelArchitecture();
        }
      }
      const isAlphaDisplay = isAlphaModelType(displayModelType);
      let displayWeights = currentWeights;
      if(isAlphaDisplay){
        displayWeights = null;
      } else if(snapshot && currentWeights){
        displayWeights = getDisplayWeightsForUi(currentWeights, { dtype: snapshot.dtype || dtypePreference });
      } else if(currentWeights){
        displayWeights = getDisplayWeightsForUi(currentWeights);
      }
      const vizFeatureNames = isAlphaDisplay ? null : featureNamesForModel(displayModelType);
      const vizInputDim = isAlphaDisplay ? null : inputDimForModel(displayModelType);
      const headlessSkip = train.enabled && train.visualizeBoard === false && !snapshot;
      const skipNetworkViz = isAlphaDisplay ? false : (headlessSkip || !displayUsesPopulation);
      if(!skipNetworkViz){
        if(isAlphaDisplay){
          const alphaState = ensureAlphaState();
          const alphaTraining = alphaState && alphaState.training ? alphaState.training : null;
          const liveSummary = alphaTraining && alphaTraining.lastConvSummary ? alphaTraining.lastConvSummary : null;
          const liveMetrics = alphaTraining && alphaTraining.lastConvMetrics ? alphaTraining.lastConvMetrics : null;
          const liveStep = alphaTraining && Number.isFinite(alphaTraining.lastConvSummaryStep)
            ? alphaTraining.lastConvSummaryStep
            : null;
          const summary = snapshot && snapshot.convSummary ? snapshot.convSummary : liveSummary;
          const metrics = snapshot && snapshot.metrics ? snapshot.metrics : liveMetrics;
          const step = snapshot && Number.isFinite(snapshot.step) ? snapshot.step : liveStep;
          if(summary){
            renderAlphaConvFilters(summary, { step, metrics });
          } else {
            renderAlphaNetworkPlaceholder('ConvNet filters will appear after a few training updates.');
          }
        } else {
          try {
            renderNetworkD3(displayWeights, overrideLayers, { featureNames: vizFeatureNames, inputDim: vizInputDim });
          } catch (_) {
            /* ignore render failures */
          }
        }
      } else if(isAlphaDisplay){
        renderAlphaNetworkPlaceholder();
      } else if(!displayUsesPopulation && networkVizEl){
        networkVizEl.innerHTML = '';
      }
    }

    function randn(){
      if(randnSpare !== null){
        const cached = randnSpare;
        randnSpare = null;
        return cached;
      }
      let u=0,v=0; while(u===0) u=Math.random(); while(v===0) v=Math.random();
      const mag = Math.sqrt(-2*Math.log(u));
      const angle = 2*Math.PI*v;
      const sample = mag*Math.cos(angle);
      randnSpare = mag*Math.sin(angle);
      return sample;
    }
    function samplePopulation(){
      if(!usesPopulationModel(train.modelType)){
        train.candWeights = [];
        train.candWeightViews = [];
        train.candScores = [];
        train.candIndex = -1;
        return;
      }
      const dim = paramDim();
      train.candWeights = [];
      train.candWeightViews = [];
      const useLowPrecisionView = dtypePreference === 'f16' && HAS_F16;
      for(let i = 0; i < train.popSize; i++){
        const master = allocFloat32(dim);
        for(let d = 0; d < dim; d++){
          const mean = (train.mean && Number.isFinite(train.mean[d])) ? train.mean[d] : 0;
          const rawStd = (train.std && Number.isFinite(train.std[d])) ? train.std[d] : 0;
          const scale = Math.max(train.minStd, Math.abs(rawStd));
          master[d] = mean + randn() * scale;
        }
        let view = master;
        if(useLowPrecisionView){
          view = new Float16Array(dim);
          copyValues(master, view);
        }
        train.candWeights.push(master);
        train.candWeightViews.push(view);
      }
      // Ensure the very first attempt (gen 0, cand 0) uses the mean weights (intentionally poor)
      if(train.gen === 0 && train.candWeights.length > 0){
        const dim0 = train.mean.length;
        const masterMean = allocFloat32(dim0);
        copyValues(train.mean, masterMean);
        let view = masterMean;
        if(dtypePreference === 'f16' && HAS_F16){
          view = new Float16Array(dim0);
          copyValues(masterMean, view);
        }
        train.candWeights[0] = masterMean;
        train.candWeightViews[0] = view;
      }
      train.candScores = new Array(train.popSize).fill(0);
      train.candIndex = 0;
    }

    function hasExistingTrainingProgress(){
      if(!train){
        return false;
      }
      if(Number.isFinite(train.gen) && train.gen > 0){
        return true;
      }
      if(Array.isArray(train.bestByGeneration) && train.bestByGeneration.length > 0){
        return true;
      }
      if(Array.isArray(train.gameScores) && train.gameScores.length > 0){
        return true;
      }
      if(Number.isFinite(train.totalGamesPlayed) && train.totalGamesPlayed > 0){
        return true;
      }
      if(Array.isArray(train.candWeights) && train.candWeights.length > 0){
        const idx = Number.isFinite(train.candIndex) ? Math.floor(train.candIndex) : -1;
        if(idx >= 0 && idx < train.candWeights.length){
          return true;
        }
        if(train.candWeights.length === train.popSize){
          return true;
        }
      }
      return false;
    }

    function startTraining(){
      if(!state.running){ start(); }
      trainingProfiler.reset();
      trainingProfiler.enable();
      const continuing = hasExistingTrainingProgress();
      const populationModel = usesPopulationModel(train.modelType);
      if(!continuing){
        train.performanceSummary = [];
        train.gen = 0;
        train.gameScores = [];
        train.gameModelTypes = [];
        train.gameScoresOffset = 0;
        train.totalGamesPlayed = 0;
        const bestCount = Array.isArray(train.bestByGeneration) ? train.bestByGeneration.length : 0;
        const genCount = Number.isFinite(train.gen) ? train.gen : 0;
        const popBaseline = Math.max(1, Number.isFinite(train.popSize) ? train.popSize : 0);
        const gameBaseline = Math.max(10, Number.isFinite(train.totalGamesPlayed) ? train.totalGamesPlayed : 0);
        const baselineSource = train.plotBestOnly
          ? Math.max(1, bestCount, genCount, populationModel ? 1 : 10)
          : (populationModel ? popBaseline : gameBaseline);
        const baselineAxis = Math.max(10, Math.ceil(baselineSource * 1.2));
        const cap = Math.max(1, train.maxPlotPoints || baselineAxis);
        train.scorePlotAxisMax = Math.min(cap, baselineAxis);
      }
      train.enabled = true;
      resetAiPlanState();
      train.ai.acc = 0;
      if(populationModel){
        if(!continuing){
          samplePopulation();
        } else {
          const needsPopulation = !Array.isArray(train.candWeights) || train.candWeights.length !== train.popSize || train.candWeights.length === 0;
          if(needsPopulation){
            samplePopulation();
          } else {
            const maxIndex = Math.max(0, train.candWeights.length - 1);
            const safeIndex = Number.isFinite(train.candIndex) ? Math.max(0, Math.min(maxIndex, Math.floor(train.candIndex))) : 0;
            train.candIndex = safeIndex;
          }
          if(!Array.isArray(train.candScores) || train.candScores.length !== train.popSize){
            train.candScores = new Array(train.popSize).fill(0);
          }
        }
      } else {
        train.candWeights = [];
        train.candWeightViews = [];
        train.candScores = [];
        train.candIndex = -1;
      }
      train.currentWeightsOverride = null;
      train.scorePlotPending = 0;
      updateTrainStatus();
      const btn = document.getElementById('start-training');
      if(btn){
        btn.textContent = 'Stop';
        btn.classList.remove('icon-btn--violet');
        btn.classList.add('icon-btn--emerald');
        btn.setAttribute('title', 'Stop training');
        btn.setAttribute('aria-label', 'Stop training');
      }
      if(continuing){
        if(populationModel){
          const totalCandidates = Number.isFinite(train.popSize) && train.popSize > 0
            ? train.popSize
            : ((Array.isArray(train.candWeights) && train.candWeights.length > 0) ? train.candWeights.length : 1);
          const candidateNumber = Math.max(
            1,
            Math.min(totalCandidates, (Number.isFinite(train.candIndex) ? Math.floor(train.candIndex) : 0) + 1)
          );
          log(`Training resumed — Gen ${train.gen + 1}, Candidate ${candidateNumber}/${totalCandidates}`);
        } else {
          log('AlphaTetris training resumed');
        }
      } else {
        log(populationModel ? 'Training started' : 'AlphaTetris training started');
      }
    }
    function stopTraining(){
      const populationModel = usesPopulationModel(train.modelType);
      const wasRunning = train.enabled;
      train.enabled = false;
      resetAiPlanState();
      const btn = document.getElementById('start-training');
      if(btn){
        btn.textContent = 'Start';
        btn.classList.remove('icon-btn--emerald');
        btn.classList.add('icon-btn--violet');
        btn.setAttribute('title', 'Start training');
        btn.setAttribute('aria-label', 'Start training');
      }
      if(train.scorePlotPending && train.gameScores.length){
        updateScorePlot();
      }
      train.scorePlotPending = 0;
      log(populationModel ? 'Training stopped' : 'AlphaTetris training stopped');
      if(wasRunning){
        logTrainingProfileSummary();
      }
      trainingProfiler.disable();
      trainingProfiler.reset();
      updateTrainStatus();
    }
    function resetTraining(){
      stopTraining();
      currentModelType = train.modelType;
      train.mlpHiddenLayers = mlpHiddenLayers.slice();
      const populationModel = usesPopulationModel(train.modelType);
      if(populationModel){
        // Reset mean/std based on selected model
        train.mean = initialMean(train.modelType);
        train.std = initialStd(train.modelType);
        train.meanView = createDisplayView(train.mean, train.meanView);
        train.stdView = createDisplayView(train.std, train.stdView);
        if(train.alpha){
          disposeAlphaModel(train.alpha);
        }
        train.alpha = null;
      } else {
        train.mean = null;
        train.std = null;
        train.meanView = null;
        train.stdView = null;
        const prevAlpha = train.alpha || null;
        const prevConfig = prevAlpha && prevAlpha.config ? prevAlpha.config : null;
        const prevModel = prevAlpha && prevAlpha.latestModel ? prevAlpha.latestModel : null;
        const prevPromise = prevAlpha && prevAlpha.modelPromise ? prevAlpha.modelPromise : null;
        const prevRootValue = prevAlpha && Number.isFinite(prevAlpha.lastRootValue)
          ? prevAlpha.lastRootValue
          : 0;
        const prevPolicy = prevAlpha && prevAlpha.lastPolicyLogits ? prevAlpha.lastPolicyLogits : null;
        train.alpha = createAlphaState(prevConfig);
        train.alpha.latestModel = prevModel || null;
        train.alpha.modelPromise = prevPromise || (prevModel ? Promise.resolve(prevModel) : null);
        train.alpha.lastRootValue = prevRootValue;
        train.alpha.lastPolicyLogits = prevPolicy;
      }
      train.gen = 0;
      train.candWeights = [];
      train.candWeightViews = [];
      train.candScores = [];
      train.candIndex = -1;
      resetAiPlanState();
      train.ai.acc = 0;
      train.performanceSummary = [];
      train.gameScores = [];
      train.gameModelTypes = [];
      train.gameScoresOffset = 0;
      train.totalGamesPlayed = 0;
      train.currentWeightsOverride = null;
      train.bestEverFitness = -Infinity;
      train.bestEverWeights = null;
      train.bestByGeneration = [];
      train.historySelection = null;
      train.scorePlotPending = 0;
      train.plotBestOnly = populationModel && !train.visualizeBoard;
      train.scorePlotUpdateFreq = populationModel
        ? SCORE_PLOT_DEFAULT_UPDATE_FREQ
        : SCORE_PLOT_ALPHATETRIS_UPDATE_FREQ;
      {
        const bestCount = Array.isArray(train.bestByGeneration) ? train.bestByGeneration.length : 0;
        const genCount = Number.isFinite(train.gen) ? train.gen : 0;
        const popBaseline = Math.max(1, Number.isFinite(train.popSize) ? train.popSize : 0);
        const gameBaseline = Math.max(10, Number.isFinite(train.totalGamesPlayed) ? train.totalGamesPlayed : 0);
        const baselineSource = train.plotBestOnly
          ? Math.max(1, bestCount, genCount, populationModel ? 1 : 10)
          : (populationModel ? popBaseline : gameBaseline);
        const baselineAxis = Math.max(10, Math.ceil(baselineSource * 1.2));
        const cap = Math.max(1, train.maxPlotPoints || baselineAxis);
        train.scorePlotAxisMax = Math.min(cap, baselineAxis);
      }
      updateScorePlot();
      syncHistoryControls();
      syncMctsControls();
      updateTrainStatus();
      log(populationModel ? 'Training parameters reset' : 'AlphaTetris training state reset');
    }
    window.startTraining = startTraining; window.stopTraining = stopTraining; window.resetTraining = resetTraining;

    function resetAiPlanState(){
      if(!train || !train.ai){
        return;
      }
      train.ai.plan = null;
      train.ai.staleMs = 0;
      train.ai.lastSig = '';
      train.ai.lastSearchStats = null;
    }

    function maybeEndForLevelCap(logPrefix = 'AI'){
      if(state.level < LEVEL_CAP){
        return false;
      }
      resetAiPlanState();
      const prefix = logPrefix ? `${logPrefix}: ` : '';
      log(`${prefix}level cap reached -> game over`);
      onGameOver();
      return true;
    }

    function onGameOver(){
      log('Game over. Resetting.');
      if(train.enabled){
        const fitness = state.score;
        const populationModel = usesPopulationModel(train.modelType);
        if(populationModel && Array.isArray(train.candScores) && train.candIndex >= 0){
          train.candScores[train.candIndex] = fitness;
        }
        // Append raw score for progress plot and mark model type
        train.gameScores.push(state.score);
        train.gameModelTypes.push(train.modelType);
        if(!Number.isFinite(train.totalGamesPlayed)){
          train.totalGamesPlayed = 0;
        }
        if(!Number.isFinite(train.gameScoresOffset)){
          train.gameScoresOffset = 0;
        }
        train.totalGamesPlayed += 1;
        // Cap data arrays to bound memory
        const cap = train.maxPlotPoints;
        const overflow = train.gameScores.length - cap;
        if (overflow > 0) {
          train.gameScores.splice(0, overflow);
          train.gameModelTypes.splice(0, overflow);
          train.gameScoresOffset += overflow;
        }
        if(train.plotBestOnly){
          train.scorePlotPending = 0;
        } else {
          const defaultStride = usesPopulationModel(train.modelType)
            ? SCORE_PLOT_DEFAULT_UPDATE_FREQ
            : SCORE_PLOT_ALPHATETRIS_UPDATE_FREQ;
          const updateStride = Math.max(
            1,
            train.scorePlotUpdateFreq || defaultStride,
          );
          train.scorePlotPending = (train.scorePlotPending || 0) + 1;
          if(train.scorePlotPending >= updateStride){
            updateScorePlot();
          }
        }
        if(!populationModel){
          if(Number.isFinite(fitness) && fitness > (train.bestEverFitness ?? -Infinity)){
            train.bestEverFitness = fitness;
          }
          Object.assign(state,{grid:emptyGrid(),active:null,next:null,score:0, level:0, pieces:0});
          state.gravity = gravityForLevel(0);
          updateLevel(); updateScore();
          spawn(); resetAiPlanState(); train.ai.acc = 0;
          updateTrainStatus();
          return;
        }
        if(train.candIndex + 1 < train.popSize){
          train.candIndex += 1;
          Object.assign(state,{grid:emptyGrid(),active:null,next:null,score:0, level:0, pieces:0});
          state.gravity = gravityForLevel(0);
          updateLevel(); updateScore();
          spawn(); resetAiPlanState(); train.ai.acc = 0;
          log(`Candidate ${train.candIndex+1}/${train.popSize} (gen ${train.gen+1})`);
          updateTrainStatus();
        } else {
          if(!Array.isArray(train.candScores) || train.candScores.length === 0){
            log('Candidate score buffer missing or empty. Reinitializing population.');
            samplePopulation();
            train.candIndex = 0;
            Object.assign(state,{grid:emptyGrid(),active:null,next:null,score:0, level:0, pieces:0});
            state.gravity = gravityForLevel(0);
            updateLevel(); updateScore();
            spawn(); resetAiPlanState(); train.ai.acc = 0;
            updateScorePlot();
            log(`Candidate ${train.candIndex+1}/${train.popSize} (gen ${train.gen+1})`);
            updateTrainStatus();
            return;
          }
          const idx = [...train.candScores.keys()].sort((a,b)=>train.candScores[b]-train.candScores[a]);
          const bestIdx = idx[0];
          const bestThisGen = train.candScores[bestIdx];
          const eliteCount = Math.max(1, Math.floor(train.eliteFrac * train.popSize));
          const elites = idx.slice(0, eliteCount);
          const dim = paramDim();
          const newMean = allocFloat32(dim);
          if(elites.length > 0){
            for(let i = 0; i < elites.length; i++){
              const wCand = train.candWeights[elites[i]];
              for(let d = 0; d < dim; d++){
                newMean[d] += wCand[d];
              }
            }
            for(let d = 0; d < dim; d++){
              newMean[d] /= elites.length;
            }
          }
          const newStd = allocFloat32(dim);
          if(elites.length > 0){
            for(let i = 0; i < elites.length; i++){
              const wCand = train.candWeights[elites[i]];
              for(let d = 0; d < dim; d++){
                const diff = wCand[d] - newMean[d];
                newStd[d] += diff * diff;
              }
            }
          }
          for(let d = 0; d < dim; d++){
            const variance = elites.length > 0 ? newStd[d] / Math.max(1, elites.length) : 0;
            const stdValue = Math.sqrt(Math.max(0, variance));
            const bounded = Math.min(train.maxStd, Math.max(train.minStd, stdValue));
            newStd[d] = Number.isFinite(bounded) ? bounded : train.minStd;
          }
          train.bestFitness = bestThisGen;
          const bestWeights = train.candWeights[bestIdx];
          const snapshot = {
            gen: train.gen + 1,
            fitness: bestThisGen,
            modelType: train.modelType,
            dtype: train.dtype,
            layerSizes: isMlpModelType(train.modelType) ? currentMlpLayerSizes(train.modelType) : [FEAT_DIM, 1],
            weights: bestWeights ? cloneWeightsArray(bestWeights) : null,
            scoreIndex: Math.max(0, train.totalGamesPlayed - train.popSize + bestIdx),
          };
          train.mean = newMean;
          train.meanView = createDisplayView(train.mean, train.meanView);
          train.std = newStd;
          train.stdView = createDisplayView(train.std, train.stdView);
          if(bestWeights && Number.isFinite(bestThisGen) && bestThisGen > (train.bestEverFitness ?? -Infinity)){
            train.bestEverFitness = bestThisGen;
            train.bestEverWeights = cloneWeightsArray(bestWeights);
          }
          recordGenerationSnapshot(snapshot);
          train.gen += 1;
          train.currentWeightsOverride = null;
          log(`Gen ${train.gen} complete. Best score: ${bestThisGen}`);
          samplePopulation();
          Object.assign(state,{grid:emptyGrid(),active:null,next:null,score:0, level:0, pieces:0});
          state.gravity = gravityForLevel(0);
          updateLevel(); updateScore();
          spawn(); resetAiPlanState(); train.ai.acc = 0;
          updateScorePlot();
          log(`Candidate ${train.candIndex+1}/${train.popSize} (gen ${train.gen+1})`);
          updateTrainStatus();
          return;
        }
      } else {
        Object.assign(state,{grid:emptyGrid(),active:null,next:null,score:0, level:0, pieces:0});
        state.gravity = gravityForLevel(0);
        updateLevel(); updateScore();
        spawn();
      }
    }
    window.__onGameOver = onGameOver;

    function stateWidth(state){ let maxC=0; for(const [,c] of state) if(c>maxC) maxC=c; return maxC+1; }
    function resetGridScratchFrom(source){
      for(let r=0; r<HEIGHT; r++){
        const srcRow = source[r];
        const dstRow = gridScratch[r];
        if(srcRow && srcRow.length){
          const limit = Math.min(WIDTH, srcRow.length);
          for(let c=0; c<limit; c++){
            dstRow[c] = srcRow[c];
          }
          for(let c=limit; c<WIDTH; c++){
            dstRow[c] = 0;
          }
        } else {
          for(let c=0; c<WIDTH; c++){
            dstRow[c] = 0;
          }
        }
      }
      return gridScratch;
    }
    function landingRowForPlacement(shape, rot, col, baselineHeights){
      if(!baselineHeights || baselineHeights.length < WIDTH){
        return null;
      }
      const states = SHAPES[shape];
      if(!states || !states.length){
        return null;
      }
      const len = states.length;
      let rotIdx = Number.isFinite(rot) ? rot : 0;
      rotIdx %= len;
      if(rotIdx < 0){
        rotIdx += len;
      }
      const rotations = pieceBottomProfiles[shape];
      if(!rotations || rotations.length <= rotIdx){
        return null;
      }
      const profile = rotations[rotIdx];
      if(!profile || profile.length === 0){
        return null;
      }
      let landingRow = HEIGHT;
      let hasColumn = false;
      for(let i = 0; i < profile.length; i += 1){
        const entry = profile[i];
        if(!entry || !Number.isFinite(entry.bottom)){
          continue;
        }
        const boardCol = col + entry.col;
        if(boardCol < 0 || boardCol >= WIDTH){
          return null;
        }
        const baseHeight = baselineHeights[boardCol] || 0;
        const stackTopRow = HEIGHT - baseHeight - 1;
        if(!Number.isFinite(stackTopRow)){
          return null;
        }
        const allowed = stackTopRow - entry.bottom;
        if(allowed < 0){
          return null;
        }
        if(allowed < landingRow){
          landingRow = allowed;
        }
        hasColumn = true;
      }
      if(!hasColumn || landingRow === HEIGHT){
        return null;
      }
      return landingRow;
    }

    function enumeratePlacements(grid, shape){
      return trainingProfiler.section('train.full.enumerate', () => {
        const actions = [];
        const rotIdx = UNIQUE_ROTATIONS[shape];
        for(const rot of rotIdx){
          const width = stateWidth(SHAPES[shape][rot]);
          for(let col=0; col<=WIDTH-width; col++){
            actions.push({ rot, col });
          }
        }
        return actions;
      });
    }
    function lockSim(grid, piece){
      const shape = piece && typeof piece.shape === 'string' ? piece.shape : 1;
      for(const [r,c] of piece.blocks()){
        grid[r][c] = shape;
      }
    }
    function clearLinesInScratch(grid, clearedRows){
      if(clearedRows){
        clearedRows.length = 0;
      }
      let write = HEIGHT - 1;
      let cleared = 0;
      for(let row = HEIGHT - 1; row >= 0; row--){
        const src = grid[row];
        let filled = true;
        for(let col = 0; col < WIDTH; col++){
          if(!src[col]){ filled = false; break; }
        }
        if(!filled){
          if(write !== row){
            const dst = grid[write];
            for(let col = 0; col < WIDTH; col++) dst[col] = src[col];
          }
          write -= 1;
        } else {
          cleared += 1;
          if(clearedRows){
            clearedRows.push(row);
          }
        }
      }
      for(let row = write; row >= 0; row--){
        const dst = grid[row];
        for(let col = 0; col < WIDTH; col++) dst[col] = 0;
      }
      return cleared;
    }
    function createEmptyRow(){
      const row = new Array(WIDTH);
      for(let col = 0; col < WIDTH; col += 1){
        row[col] = 0;
      }
      return row;
    }
    function applyClearedRowsToGrid(grid, clearedRows){
      if(!grid || !Array.isArray(clearedRows) || clearedRows.length === 0){
        return 0;
      }
      const sorted = clearedRows.slice().sort((a,b) => b - a);
      let cleared = 0;
      for(let i = 0; i < sorted.length; i += 1){
        const rawIndex = sorted[i];
        const rowIndex = Number.isFinite(rawIndex) ? Math.floor(rawIndex) : -1;
        if(rowIndex < 0 || rowIndex >= grid.length){
          continue;
        }
        grid.splice(rowIndex, 1);
        cleared += 1;
      }
      for(let i = 0; i < cleared; i += 1){
        grid.unshift(createEmptyRow());
      }
      while(grid.length > HEIGHT){
        grid.pop();
      }
      while(grid.length < HEIGHT){
        grid.unshift(createEmptyRow());
      }
      return cleared;
    }
    const BOARD_AREA = WIDTH * HEIGHT;
    const BUMP_NORMALIZER = ((WIDTH - 1) * HEIGHT) || 1;
    const CONTACT_NORMALIZER = (BOARD_AREA * 2) || 1;
    const ENGINEERED_FEATURE_INDEX = {
      HOLE_RATIO: 6,
      NEW_HOLE_RATIO: 7,
      BUMP_RATIO: 8,
      MAX_HEIGHT_RATIO: 9,
      WELL_SUM_RATIO: 10,
      EDGE_WELL_RATIO: 11,
      TETRIS_WELL_RATIO: 12,
      CONTACT_RATIO: 13,
      ROW_TRANSITION_RATIO: 14,
      COL_TRANSITION_RATIO: 15,
      AGG_HEIGHT_RATIO: 16,
    };
    // Keep direct line-clear rewards small so structural heuristics dominate exploration.
    const IMMEDIATE_REWARD_SCALE = 0.1;
    const REWARD_SHAPING_WEIGHTS = Object.freeze({
      survival: 0.02,
      topOutPenalty: 1,
      newHolePenalty: 0.25,
      holeReduction: 0.25,
      bumpReduction: 0.02,
      aggregateHeightReduction: 0.02,
      aggregateHeightIncreasePenalty: 0.03,
      maxHeightReduction: 0.05,
      maxHeightIncreasePenalty: 0.08,
      contactIncrease: 0.001,
      rowTransitionReduction: 0.02,
      colTransitionReduction: 0.02,
      tetrisWellBonus: 0.05,
      edgeWellPenalty: 0.05,
      singlePenalty: 0.02,
      doubleBonus: 0.05,
      tripleBonus: 0.1,
      tetrisBonus: 0.2,
    });
    function engineeredFeatureValue(features, index, scale = 1){
      if(!features || features.length <= index){
        return 0;
      }
      const raw = features[index];
      return Number.isFinite(raw) ? raw * scale : 0;
    }
    function computePlacementReward({ lines = 0, features = null, baselineFeatures = null, newHoleCount = 0, topOut = false }){
      const baseReward = Number.isFinite(lines) ? lines / 4 : 0;
      const scaledBaseReward = baseReward * IMMEDIATE_REWARD_SCALE;
      let shaped = scaledBaseReward;
      if(topOut){
        shaped -= REWARD_SHAPING_WEIGHTS.topOutPenalty;
      } else {
        shaped += REWARD_SHAPING_WEIGHTS.survival;
      }
      if(Number.isFinite(newHoleCount) && newHoleCount > 0){
        shaped -= newHoleCount * REWARD_SHAPING_WEIGHTS.newHolePenalty;
      }
      if(
        features &&
        baselineFeatures &&
        features.length > ENGINEERED_FEATURE_INDEX.AGG_HEIGHT_RATIO &&
        baselineFeatures.length > ENGINEERED_FEATURE_INDEX.AGG_HEIGHT_RATIO
      ){
        const holeReduction = engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.HOLE_RATIO, BOARD_AREA)
          - engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.HOLE_RATIO, BOARD_AREA);
        if(holeReduction > 0){
          shaped += holeReduction * REWARD_SHAPING_WEIGHTS.holeReduction;
        }
        const bumpReduction = engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.BUMP_RATIO, BUMP_NORMALIZER)
          - engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.BUMP_RATIO, BUMP_NORMALIZER);
        if(bumpReduction > 0){
          shaped += bumpReduction * REWARD_SHAPING_WEIGHTS.bumpReduction;
        }
        const baselineAggregateHeight = engineeredFeatureValue(
          baselineFeatures,
          ENGINEERED_FEATURE_INDEX.AGG_HEIGHT_RATIO,
          BOARD_AREA
        );
        const newAggregateHeight = engineeredFeatureValue(
          features,
          ENGINEERED_FEATURE_INDEX.AGG_HEIGHT_RATIO,
          BOARD_AREA
        );
        const aggregateHeightDelta = newAggregateHeight - baselineAggregateHeight;
        if(aggregateHeightDelta < 0){
          shaped += (-aggregateHeightDelta) * REWARD_SHAPING_WEIGHTS.aggregateHeightReduction;
        } else if(aggregateHeightDelta > 0){
          shaped -= aggregateHeightDelta * REWARD_SHAPING_WEIGHTS.aggregateHeightIncreasePenalty;
        }
        const baselineMaxHeight = engineeredFeatureValue(
          baselineFeatures,
          ENGINEERED_FEATURE_INDEX.MAX_HEIGHT_RATIO,
          HEIGHT
        );
        const newMaxHeight = engineeredFeatureValue(
          features,
          ENGINEERED_FEATURE_INDEX.MAX_HEIGHT_RATIO,
          HEIGHT
        );
        const maxHeightDelta = newMaxHeight - baselineMaxHeight;
        if(maxHeightDelta < 0){
          shaped += (-maxHeightDelta) * REWARD_SHAPING_WEIGHTS.maxHeightReduction;
        } else if(maxHeightDelta > 0){
          shaped -= maxHeightDelta * REWARD_SHAPING_WEIGHTS.maxHeightIncreasePenalty;
        }
        const contactIncrease = engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.CONTACT_RATIO, CONTACT_NORMALIZER)
          - engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.CONTACT_RATIO, CONTACT_NORMALIZER);
        if(contactIncrease > 0){
          shaped += contactIncrease * REWARD_SHAPING_WEIGHTS.contactIncrease;
        }
        const rowTransitionReduction = engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.ROW_TRANSITION_RATIO, BOARD_AREA)
          - engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.ROW_TRANSITION_RATIO, BOARD_AREA);
        if(rowTransitionReduction > 0){
          shaped += rowTransitionReduction * REWARD_SHAPING_WEIGHTS.rowTransitionReduction;
        }
        const colTransitionReduction = engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.COL_TRANSITION_RATIO, BOARD_AREA)
          - engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.COL_TRANSITION_RATIO, BOARD_AREA);
        if(colTransitionReduction > 0){
          shaped += colTransitionReduction * REWARD_SHAPING_WEIGHTS.colTransitionReduction;
        }
        const tetrisWellGain = engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.TETRIS_WELL_RATIO, HEIGHT)
          - engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.TETRIS_WELL_RATIO, HEIGHT);
        if(tetrisWellGain > 0){
          shaped += tetrisWellGain * REWARD_SHAPING_WEIGHTS.tetrisWellBonus;
        }
        const edgeWellIncrease = engineeredFeatureValue(features, ENGINEERED_FEATURE_INDEX.EDGE_WELL_RATIO, HEIGHT)
          - engineeredFeatureValue(baselineFeatures, ENGINEERED_FEATURE_INDEX.EDGE_WELL_RATIO, HEIGHT);
        if(edgeWellIncrease > 0){
          shaped -= edgeWellIncrease * REWARD_SHAPING_WEIGHTS.edgeWellPenalty;
        }
      }
      if(lines === 1){
        shaped -= REWARD_SHAPING_WEIGHTS.singlePenalty * IMMEDIATE_REWARD_SCALE;
      } else if(lines === 2){
        shaped += REWARD_SHAPING_WEIGHTS.doubleBonus * IMMEDIATE_REWARD_SCALE;
      } else if(lines === 3){
        shaped += REWARD_SHAPING_WEIGHTS.tripleBonus * IMMEDIATE_REWARD_SCALE;
      } else if(lines >= 4){
        shaped += REWARD_SHAPING_WEIGHTS.tetrisBonus * IMMEDIATE_REWARD_SCALE;
      }
      return shaped;
    }
    function wellMetrics(heights){
      let wellSum=0;
      let edgeWell=0;
      let maxWellDepth=0;
      let wellCount=0;
      for(let c=0;c<WIDTH;c++){
        const left = (c>0)?heights[c-1]:Infinity;
        const right = (c<WIDTH-1)?heights[c+1]:Infinity;
        const minNbr = Math.min(left,right);
        const depth = minNbr - heights[c];
        if(depth>0){
          wellSum += depth;
          wellCount++;
          if(depth>maxWellDepth){
            maxWellDepth = depth;
          }
        }
        if(c===0){
          edgeWell = Math.max(edgeWell, right - heights[0]);
        }
        if(c===WIDTH-1){
          edgeWell = Math.max(edgeWell, left - heights[WIDTH-1]);
        }
      }
      const safeEdge = Math.max(0, edgeWell);
      const tetrisWell = (wellCount === 1) ? maxWellDepth : 0;
      return {wellSum, edgeWell: safeEdge, maxWellDepth, wellCount, tetrisWell};
    }
    function simulateAfterPlacement(grid, shape, rot, col){
      return trainingProfiler.section('train.full.simulate', () => {
        const g = resetGridScratchFrom(grid);
        const piece = pooledPiece;
        piece.shape = shape;
        piece.rot = rot;
        piece.row = 0;
        piece.col = col;
        const dropRow = landingRowForPlacement(shape, piece.rot, col, baselineColumnHeightScratch);
        if(dropRow === null) return null;
        piece.row = dropRow;
        for(const [r,c] of piece.blocks()){
          if(r < 0 || r >= HEIGHT || c < 0 || c >= WIDTH || g[r][c]){
            return null;
          }
        }
        lockSim(g, piece);
        const lines = clearLinesInScratch(g, clearedRowsScratch);
        simulateResultScratch.lines = lines;
        simulateResultScratch.grid = g;
        simulateResultScratch.dropRow = dropRow;
        simulateResultScratch.clearedRows = clearedRowsScratch;
        simulateResultScratch.clearedRowCount = clearedRowsScratch.length;
        return simulateResultScratch;
      });
    }
    function resetMetricsObject(metrics){
      metrics.holes = 0;
      metrics.bump = 0;
      metrics.maxHeight = 0;
      metrics.wellSum = 0;
      metrics.edgeWell = 0;
      metrics.tetrisWell = 0;
      metrics.contact = 0;
      metrics.rowTransitions = 0;
      metrics.colTransitions = 0;
      metrics.aggregateHeight = 0;
      return metrics;
    }
    function fillRowMaskFromGrid(grid, target){
      if(!target){
        return null;
      }
      for(let r = 0; r < HEIGHT; r += 1){
        const row = grid && grid[r];
        let mask = 0;
        if(row && row.length){
          for(let c = 0; c < WIDTH; c += 1){
            if(row[c]){
              mask |= 1 << c;
            }
          }
        }
        target[r] = mask;
      }
      return target;
    }
    function placementSurrogateKeyFor(shape, rot, col, baselineMasks, baselineHeights){
      if(!shape || !SHAPES[shape]){
        return null;
      }
      const rotations = pieceBottomProfiles[shape];
      if(!rotations || !rotations.length){
        return null;
      }
      const len = rotations.length;
      let rotIdx = Number.isFinite(rot) ? rot : 0;
      rotIdx %= len;
      if(rotIdx < 0){
        rotIdx += len;
      }
      const profile = rotations[rotIdx];
      if(!profile || !profile.length){
        return null;
      }
      const parts = [`${shape}:${rotIdx}:${col}`];
      for(let i = 0; i < profile.length; i += 1){
        const entry = profile[i];
        if(!entry){
          parts.push('');
          continue;
        }
        const boardCol = col + entry.col;
        if(boardCol < 0 || boardCol >= WIDTH){
          return null;
        }
        const mask = baselineMasks && baselineMasks.length > boardCol ? baselineMasks[boardCol] || 0 : 0;
        const height = baselineHeights && baselineHeights.length > boardCol ? baselineHeights[boardCol] || 0 : 0;
        parts.push(`${boardCol.toString(36)},${mask.toString(36)},${height.toString(36)}`);
      }
      return parts.join(';');
    }
    function decodePlacementSurrogateKey(key){
      if(typeof key !== 'string' || !key){
        return null;
      }
      const cached = placementSurrogateKeyCache.get(key);
      if(cached){
        return cached;
      }
      const segments = key.split(';');
      if(!segments.length){
        return null;
      }
      const header = segments[0] || '';
      const headerParts = header.split(':');
      if(headerParts.length < 3){
        return null;
      }
      const shape = headerParts[0];
      const rot = Number.parseInt(headerParts[1], 10);
      const col = Number.parseInt(headerParts[2], 10);
      const touched = [];
      for(let i = 1; i < segments.length; i += 1){
        const seg = segments[i];
        if(!seg){
          continue;
        }
        const parts = seg.split(',');
        if(parts.length < 3){
          continue;
        }
        const boardCol = Number.parseInt(parts[0], 36);
        const maskRaw = Number.parseInt(parts[1], 36);
        const heightRaw = Number.parseInt(parts[2], 36);
        if(!Number.isFinite(boardCol)){
          continue;
        }
        touched.push({
          col: boardCol,
          mask: Number.isFinite(maskRaw) ? maskRaw >>> 0 : 0,
          height: Number.isFinite(heightRaw) ? heightRaw : 0,
        });
      }
      const decoded = {
        shape,
        rot: Number.isFinite(rot) ? rot : 0,
        col: Number.isFinite(col) ? col : 0,
        touched,
      };
      placementSurrogateKeyCache.set(key, decoded);
      return decoded;
    }
    function computeMetricsFromRowMasks(rowMasks, columnMasks, columnHeights, metrics){
      resetMetricsObject(metrics);
      if(columnMasks){
        for(let c = 0; c < WIDTH; c += 1){
          columnMasks[c] = 0;
          if(columnHeights && columnHeights.length > c){
            columnHeights[c] = 0;
          }
        }
      }
      let prevMask = 0;
      for(let r = 0; r < HEIGHT; r += 1){
        const mask = rowMasks && rowMasks.length > r ? rowMasks[r] || 0 : 0;
        metrics.rowTransitions += rowTransitionTable[mask];
        const horizontal = rowHorizontalContactTable[mask];
        if(horizontal){
          metrics.contact += horizontal;
        }
        const shared = mask & prevMask;
        if(shared){
          metrics.contact += rowPopcountTable[shared];
        }
        const diff = mask ^ prevMask;
        if(diff){
          metrics.colTransitions += rowPopcountTable[diff];
        }
        if(columnMasks){
          let bits = mask;
          while(bits){
            const lsb = bits & -bits;
            const idx = bitIndexTable[lsb];
            if(idx >= 0 && idx < WIDTH){
              columnMasks[idx] |= 1 << (HEIGHT - 1 - r);
            }
            bits ^= lsb;
          }
        }
        prevMask = mask;
      }
      if(prevMask){
        const bottomPop = rowPopcountTable[prevMask];
        metrics.colTransitions += bottomPop;
        metrics.contact += bottomPop;
      }
      let aggregateHeight = 0;
      let maxHeight = 0;
      let holes = 0;
      if(columnMasks){
        for(let c = 0; c < WIDTH; c += 1){
          const mask = columnMasks[c] || 0;
          if(mask){
            const topBit = 31 - Math.clz32(mask);
            const height = topBit + 1;
            if(columnHeights && columnHeights.length > c){
              columnHeights[c] = height;
            }
            aggregateHeight += height;
            if(height > maxHeight){
              maxHeight = height;
            }
            if(topBit > 0){
              const belowMask = mask & ((1 << topBit) - 1);
              const filledBelow = countBits32(belowMask);
              holes += topBit - filledBelow;
            }
          } else if(columnHeights && columnHeights.length > c){
            columnHeights[c] = 0;
          }
        }
      }
      metrics.holes = holes;
      metrics.aggregateHeight = aggregateHeight;
      metrics.maxHeight = maxHeight;
      let bump = 0;
      if(columnHeights){
        for(let c = 0; c < WIDTH - 1; c += 1){
          const a = columnHeights[c] || 0;
          const b = columnHeights[c + 1] || 0;
          bump += Math.abs(a - b);
        }
      }
      metrics.bump = bump;
      if(columnHeights){
        let { wellSum, edgeWell, tetrisWell } = wellMetrics(columnHeights);
        if(holes > 0){
          tetrisWell = 0;
        }
        metrics.wellSum = wellSum;
        metrics.edgeWell = edgeWell;
        metrics.tetrisWell = tetrisWell;
      }
      return metrics;
    }
    function computeNewHoleCountFromMasks(columnMasks, baselineMasks, clearedRows){
      if(!columnMasks || !baselineMasks){
        return 0;
      }
      let count = 0;
      for(let c = 0; c < WIDTH; c += 1){
        const finalMask = columnMasks[c] || 0;
        if(!finalMask){
          continue;
        }
        const baselineMaskRaw = baselineMasks[c] || 0;
        const baselineMask = clearedRows && clearedRows.length
          ? applyClearedRowsToMask(baselineMaskRaw, clearedRows)
          : baselineMaskRaw;
        const finalHoleMask = holeMaskForColumn(finalMask);
        if(!finalHoleMask){
          continue;
        }
        const baselineHoleMask = holeMaskForColumn(baselineMask);
        const diffMask = finalHoleMask & ~baselineHoleMask;
        if(diffMask){
          count += countBits32(diffMask);
        }
      }
      return count;
    }
    function fillRawFeatureVectorFromRowMasks(target, rowMasks){
      if(!target){
        return null;
      }
      let idx = 0;
      for(let r = 0; r < HEIGHT; r += 1){
        const mask = rowMasks && rowMasks.length > r ? rowMasks[r] || 0 : 0;
        for(let c = 0; c < WIDTH; c += 1){
          target[idx] = (mask >> c) & 1 ? 1 : 0;
          idx += 1;
        }
      }
      return target;
    }
    function estimatePlacementWithSurrogate(key, options = {}){
      const decoded = decodePlacementSurrogateKey(key);
      if(!decoded){
        return null;
      }
      const { shape, rot, col, touched } = decoded;
      const baselineRowMasks = options.baselineRowMasks;
      const baselineColumnMasks = options.baselineColumnMasks;
      const baselineColumnHeights = options.baselineColumnHeights;
      if(!baselineRowMasks || !baselineColumnMasks || !baselineColumnHeights){
        return null;
      }
      const rotations = pieceBottomProfiles[shape];
      const states = SHAPES[shape];
      if(!rotations || !rotations.length || !states || !states.length){
        return null;
      }
      const len = rotations.length;
      let rotIdx = Number.isFinite(rot) ? rot : 0;
      rotIdx %= len;
      if(rotIdx < 0){
        rotIdx += len;
      }
      const profile = rotations[rotIdx];
      const state = states[rotIdx];
      if(!profile || !profile.length || !state){
        return null;
      }
      if(!Array.isArray(touched) || touched.length !== profile.length){
        return null;
      }
      let dropRow = HEIGHT;
      for(let i = 0; i < profile.length; i += 1){
        const entry = profile[i];
        const touch = touched[i];
        if(!entry || !touch){
          return null;
        }
        const boardCol = touch.col;
        if(boardCol < 0 || boardCol >= WIDTH){
          return null;
        }
        const baselineMask = baselineColumnMasks[boardCol] || 0;
        const baselineHeight = baselineColumnHeights[boardCol] || 0;
        if(baselineMask !== touch.mask || baselineHeight !== touch.height){
          return null;
        }
        const stackTopRow = HEIGHT - baselineHeight - 1;
        const allowed = stackTopRow - entry.bottom;
        if(!Number.isFinite(allowed) || allowed < 0){
          return null;
        }
        if(allowed < dropRow){
          dropRow = allowed;
        }
      }
      if(dropRow === HEIGHT){
        return null;
      }
      const rowMasks = surrogateRowMaskScratch;
      const collapsed = surrogateRowMaskCollapsedScratch;
      for(let r = 0; r < HEIGHT; r += 1){
        rowMasks[r] = baselineRowMasks[r] || 0;
        collapsed[r] = 0;
        surrogateRowTouchedFlags[r] = 0;
      }
      const touchedRows = [];
      for(let i = 0; i < state.length; i += 1){
        const block = state[i];
        if(!block || block.length < 2){
          return null;
        }
        const finalRow = dropRow + block[0];
        const finalCol = col + block[1];
        if(finalRow < 0 || finalRow >= HEIGHT || finalCol < 0 || finalCol >= WIDTH){
          return null;
        }
        const bit = 1 << finalCol;
        if(rowMasks[finalRow] & bit){
          return null;
        }
        rowMasks[finalRow] |= bit;
        if(!surrogateRowTouchedFlags[finalRow]){
          surrogateRowTouchedFlags[finalRow] = 1;
          touchedRows.push(finalRow);
        }
      }
      const clearedRows = surrogateClearedRowsScratch;
      clearedRows.length = 0;
      for(let i = 0; i < touchedRows.length; i += 1){
        const row = touchedRows[i];
        if(rowMasks[row] === ROW_MASK_LIMIT){
          clearedRows.push(row);
          surrogateClearedRowFlags[row] = 1;
        }
        surrogateRowTouchedFlags[row] = 0;
      }
      if(clearedRows.length > 1){
        clearedRows.sort((a, b) => b - a);
      }
      let finalRows = rowMasks;
      if(clearedRows.length){
        let write = HEIGHT - 1;
        for(let r = HEIGHT - 1; r >= 0; r -= 1){
          if(surrogateClearedRowFlags[r]){
            continue;
          }
          collapsed[write] = rowMasks[r];
          write -= 1;
        }
        for(; write >= 0; write -= 1){
          collapsed[write] = 0;
        }
        for(let i = 0; i < clearedRows.length; i += 1){
          surrogateClearedRowFlags[clearedRows[i]] = 0;
        }
        finalRows = collapsed;
      }
      const metrics = computeMetricsFromRowMasks(
        finalRows,
        surrogateColumnMaskEstimateScratch,
        surrogateColumnHeightEstimateScratch,
        surrogateMetricsScratch,
      );
      const lines = clearedRows.length;
      const topOut = (finalRows[0] & ROW_MASK_LIMIT) !== 0;
      const newHoleCount = computeNewHoleCountFromMasks(
        surrogateColumnMaskEstimateScratch,
        baselineColumnMasks,
        clearedRows,
      );
      const featureBuffer = fillFeatureVector(featureScratch, lines, metrics, newHoleCount);
      const engineeredFeatures = new Float32Array(featureBuffer.length);
      engineeredFeatures.set(featureBuffer);
      let rawFeatures = null;
      if(options.needRawFeatures){
        const rawBuffer = fillRawFeatureVectorFromRowMasks(rawFeatureScratch, finalRows);
        rawFeatures = new Float32Array(rawBuffer.length);
        rawFeatures.set(rawBuffer);
      }
      const metricsSnapshot = {
        holes: metrics.holes,
        bump: metrics.bump,
        maxHeight: metrics.maxHeight,
        wellSum: metrics.wellSum,
        edgeWell: metrics.edgeWell,
        tetrisWell: metrics.tetrisWell,
        contact: metrics.contact,
        rowTransitions: metrics.rowTransitions,
        colTransitions: metrics.colTransitions,
        aggregateHeight: metrics.aggregateHeight,
      };
      const clearedCopy = clearedRows.length ? clearedRows.slice() : [];
      return {
        success: true,
        dropRow,
        lines,
        topOut,
        newHoleCount,
        engineeredFeatures,
        rawFeatures,
        clearedRows: clearedCopy,
        metrics: metricsSnapshot,
      };
    }
    function maybeLogPlacementSurrogateStats(){
      const stats = placementSurrogate.stats;
      const total = stats.successes + stats.fallbacks;
      if(total <= 0 || total < placementSurrogate.nextLogThreshold){
        return;
      }
      const missRate = total ? (stats.fallbacks / total) * 100 : 0;
      const avgSurrogate = stats.successes ? stats.timeSurrogate / stats.successes : 0;
      const avgSim = stats.fallbacks ? stats.timeFallback / stats.fallbacks : 0;
      const avgDebug = stats.comparisons ? stats.timeDebugCompare / stats.comparisons : 0;
      const ratio = avgSurrogate > 0 && avgSim > 0 ? avgSim / avgSurrogate : 0;
      if(typeof console !== 'undefined' && console.log){
        console.log(
          '[Surrogate] attempts=%d success=%d fallback=%d missRate=%s avgSur=%.3fms avgSim=%.3fms speedup=%s debugAvg=%.3fms mismatches=%d',
          total,
          stats.successes,
          stats.fallbacks,
          `${missRate.toFixed(2)}%`,
          avgSurrogate,
          avgSim,
          ratio ? `${ratio.toFixed(2)}x` : 'n/a',
          avgDebug,
          stats.mismatches,
        );
      }
      placementSurrogate.nextLogThreshold = total + 500;
    }
    function holeMaskForColumn(mask){
      if(!mask){
        return 0;
      }
      const topBit = 31 - Math.clz32(mask);
      if(topBit <= 0){
        return 0;
      }
      const rangeMask = (1 << topBit) - 1;
      return (~mask) & rangeMask;
    }
    function removeBitAtIndex(mask, bitIndex){
      if(bitIndex < 0){
        return mask;
      }
      const lower = bitIndex > 0 ? mask & ((1 << bitIndex) - 1) : 0;
      const higher = mask >>> (bitIndex + 1);
      return lower | (higher << bitIndex);
    }
    function applyClearedRowsToMask(mask, clearedRows){
      if(!mask || !clearedRows || clearedRows.length === 0){
        return mask;
      }
      let result = mask;
      let removedBelow = 0;
      for(let i = 0; i < clearedRows.length; i += 1){
        const row = clearedRows[i];
        if(row < 0 || row >= HEIGHT){
          continue;
        }
        const adjustedRow = row + removedBelow;
        if(adjustedRow < 0 || adjustedRow >= HEIGHT){
          removedBelow += 1;
          continue;
        }
        const bitIndex = HEIGHT - 1 - adjustedRow;
        result = removeBitAtIndex(result, bitIndex);
        removedBelow += 1;
      }
      return result;
    }
    function computeGridMetrics(g){
      const metrics = metricsScratch;
      metrics.holes = 0;
      metrics.bump = 0;
      metrics.maxHeight = 0;
      metrics.wellSum = 0;
      metrics.edgeWell = 0;
      metrics.tetrisWell = 0;
      metrics.contact = 0;
      metrics.rowTransitions = 0;
      metrics.colTransitions = 0;
      metrics.aggregateHeight = 0;

      const heights = columnHeightScratch;
      const columnMasks = columnMaskScratch;
      if (typeof columnMasks.fill === 'function') {
        columnMasks.fill(0);
      } else {
        for (let i = 0; i < columnMasks.length; i += 1) columnMasks[i] = 0;
      }

      let prevMask = 0;
      for (let r = 0; r < HEIGHT; r += 1) {
        const row = g[r];
        let mask = 0;
        for (let c = 0; c < WIDTH; c += 1) {
          if (row[c]) {
            mask |= 1 << c;
          }
        }
        metrics.rowTransitions += rowTransitionTable[mask];
        const horizontalContact = rowHorizontalContactTable[mask];
        if (horizontalContact) {
          metrics.contact += horizontalContact;
        }
        const shared = mask & prevMask;
        if (shared) {
          metrics.contact += rowPopcountTable[shared];
        }
        const diff = mask ^ prevMask;
        if (diff) {
          metrics.colTransitions += rowPopcountTable[diff];
        }
        let bits = mask;
        while (bits) {
          const lsb = bits & -bits;
          const idx = bitIndexTable[lsb];
          if (idx >= 0 && idx < WIDTH) {
            columnMasks[idx] |= 1 << (HEIGHT - 1 - r);
          }
          bits ^= lsb;
        }
        prevMask = mask;
      }

      if (prevMask) {
        const bottomPop = rowPopcountTable[prevMask];
        metrics.colTransitions += bottomPop;
        metrics.contact += bottomPop;
      }

      let aggregateHeight = 0;
      let maxHeight = 0;
      let holes = 0;

      for (let c = 0; c < WIDTH; c += 1) {
        const mask = columnMasks[c] || 0;
        if (mask) {
          const topBit = 31 - Math.clz32(mask);
          const height = topBit + 1;
          heights[c] = height;
          aggregateHeight += height;
          if (height > maxHeight) {
            maxHeight = height;
          }
          if (topBit > 0) {
            const belowMask = mask & ((1 << topBit) - 1);
            const filledBelow = countBits32(belowMask);
            holes += topBit - filledBelow;
          }
        } else {
          heights[c] = 0;
        }
      }

      let bump = 0;
      for (let c = 0; c < WIDTH - 1; c += 1) {
        bump += Math.abs(heights[c] - heights[c + 1]);
      }

      let { wellSum, edgeWell, tetrisWell } = wellMetrics(heights);
      if (holes > 0) {
        tetrisWell = 0;
      }

      metrics.aggregateHeight = aggregateHeight;
      metrics.maxHeight = maxHeight;
      metrics.holes = holes;
      metrics.bump = bump;
      metrics.wellSum = wellSum;
      metrics.edgeWell = edgeWell;
      metrics.tetrisWell = tetrisWell;
      return metrics;
    }
    function fillFeatureVector(target, lines, metrics, newHoles){
      let cleared = (typeof lines === 'number' && Number.isFinite(lines)) ? lines : 0;
      if(cleared < 0) cleared = 0;
      target[0] = cleared / 4;
      target[1] = (cleared * cleared) / 16;
      target[2] = cleared === 1 ? 1 : 0;
      target[3] = cleared === 2 ? 1 : 0;
      target[4] = cleared === 3 ? 1 : 0;
      target[5] = cleared === 4 ? 1 : 0;
      const area = BOARD_AREA || 1;
      const holes = metrics && typeof metrics.holes === 'number' ? metrics.holes : 0;
      const bump = metrics && typeof metrics.bump === 'number' ? metrics.bump : 0;
      const maxHeight = metrics && typeof metrics.maxHeight === 'number' ? metrics.maxHeight : 0;
      const wellSum = metrics && typeof metrics.wellSum === 'number' ? metrics.wellSum : 0;
      const edgeWell = metrics && typeof metrics.edgeWell === 'number' ? metrics.edgeWell : 0;
      const tetrisWell = metrics && typeof metrics.tetrisWell === 'number' ? metrics.tetrisWell : 0;
      const contact = metrics && typeof metrics.contact === 'number' ? metrics.contact : 0;
      const rowTransitions = metrics && typeof metrics.rowTransitions === 'number' ? metrics.rowTransitions : 0;
      const colTransitions = metrics && typeof metrics.colTransitions === 'number' ? metrics.colTransitions : 0;
      const aggregateHeight = metrics && typeof metrics.aggregateHeight === 'number' ? metrics.aggregateHeight : 0;
      const boundedNewHoles = Number.isFinite(newHoles) && newHoles > 0 ? newHoles : 0;
      target[6] = holes / area;
      target[7] = boundedNewHoles / area;
      target[8] = bump / BUMP_NORMALIZER;
      target[9] = HEIGHT ? maxHeight / HEIGHT : 0;
      target[10] = wellSum / area;
      target[11] = HEIGHT ? edgeWell / HEIGHT : 0;
      target[12] = HEIGHT ? tetrisWell / HEIGHT : 0;
      target[13] = contact / CONTACT_NORMALIZER;
      target[14] = rowTransitions / area;
      target[15] = colTransitions / area;
      target[16] = aggregateHeight / area;
      return target;
    }
    function featuresFromGrid(g, lines, options = {}){
      return trainingProfiler.section('train.full.features', () => {
        const metrics = computeGridMetrics(g);
        const baselineHoles = Number.isFinite(options.holeBaseline) ? options.holeBaseline : 0;
        const baselineColumnMasks = options && options.baselineColumnMasks;
        const clearedRows = options && options.clearedRows;
        let newHoleCount = 0;
        if (baselineColumnMasks && baselineColumnMasks.length >= WIDTH) {
          const clearedSource = Array.isArray(clearedRows) ? clearedRows : null;
          for (let c = 0; c < WIDTH; c += 1) {
            const finalMask = columnMaskScratch[c] || 0;
            if (!finalMask) {
              continue;
            }
            const baselineMaskRaw = baselineColumnMasks[c] || 0;
            const baselineMask = clearedSource && clearedSource.length ? applyClearedRowsToMask(baselineMaskRaw, clearedSource) : baselineMaskRaw;
            const finalHoleMask = holeMaskForColumn(finalMask);
            if (!finalHoleMask) {
              continue;
            }
            const baselineHoleMask = holeMaskForColumn(baselineMask);
            const diffMask = finalHoleMask & ~baselineHoleMask;
            if (diffMask) {
              newHoleCount += countBits32(diffMask);
            }
          }
        } else {
          const diff = metrics.holes - baselineHoles;
          if (diff > 0) {
            newHoleCount = diff;
          }
        }
      return fillFeatureVector(featureScratch, lines, metrics, newHoleCount);
    });
    }

    function fillRawFeatureVector(target, grid){
      let idx = 0;
      for(let r = 0; r < HEIGHT; r += 1){
        const row = grid[r];
        for(let c = 0; c < WIDTH; c += 1){
          target[idx] = row && row[c] ? 1 : 0;
          idx += 1;
        }
      }
      return target;
    }

    function rawFeaturesFromGrid(g){
      return trainingProfiler.section('train.full.features_raw', () => fillRawFeatureVector(rawFeatureScratch, g));
    }

    function dot(weights, feats){
      if(!weights || !weights.length){
        return 0;
      }
      const limit = Math.min(FEAT_DIM, weights.length);
      let s = 0;
      for(let d = 0; d < limit; d++){
        s += weights[d] * feats[d];
      }
      return s;
    }
    const mlpActivationScratch = [];
    let mlpOutputScratch = null;
    let mlpInputScratch = null;

    function getMlpLayerScratch(layerIdx, size){
      let buffer = mlpActivationScratch[layerIdx];
      if(!buffer || buffer.length !== size){
        buffer = new Float32Array(size);
        mlpActivationScratch[layerIdx] = buffer;
      } else {
        buffer.fill(0);
      }
      return buffer;
    }

    function getMlpOutputScratch(){
      if(!mlpOutputScratch || mlpOutputScratch.length !== 1){
        mlpOutputScratch = new Float32Array(1);
      } else {
        mlpOutputScratch[0] = 0;
      }
      return mlpOutputScratch;
    }

    function ensureFloat32Activations(feats, size){
      if(feats instanceof Float32Array && feats.length >= size){
        return feats;
      }
      if(!mlpInputScratch || mlpInputScratch.length !== size){
        mlpInputScratch = new Float32Array(size);
      }
      copyValues(feats, mlpInputScratch);
      return mlpInputScratch;
    }

    function mlpScore(weights, feats, modelType = train.modelType){
      if(!weights || !weights.length){
        return 0;
      }
      const resolvedType = resolveMlpType(modelType);
      const hiddenLayers = mlpHiddenLayers.length ? mlpHiddenLayers : DEFAULT_MLP_HIDDEN;
      let offset = 0;
      let prevSize = inputDimForModel(resolvedType);
      let activations = ensureFloat32Activations(feats, prevSize);
      const weightLen = weights.length;
      for(let layerIdx = 0; layerIdx < hiddenLayers.length; layerIdx++){
        const layerSize = hiddenLayers[layerIdx];
        const weightBase = offset;
        const biasBase = weightBase + prevSize * layerSize;
        const nextActivations = getMlpLayerScratch(layerIdx, layerSize);
        const weightLimit = Math.min(biasBase, weightLen);
        for(let i = 0; i < prevSize; i++){
          const value = activations[i];
          if(!Number.isFinite(value) || value === 0){
            continue;
          }
          const base = weightBase + i * layerSize;
          if(base >= weightLimit){
            break;
          }
          const maxJ = Math.min(layerSize, weightLimit - base);
          for(let j = 0; j < maxJ; j++){
            nextActivations[j] += value * weights[base + j];
          }
        }
        for(let j = 0; j < layerSize; j++){
          const biasIdx = biasBase + j;
          const sum = nextActivations[j] + (biasIdx < weightLen ? weights[biasIdx] : 0);
          nextActivations[j] = sum > 0 ? sum : 0;
        }
        offset = biasBase + layerSize;
        activations = nextActivations;
        prevSize = layerSize;
      }
      const outWeightsBase = offset;
      const outBiasIndex = outWeightsBase + prevSize;
      const outLimit = Math.min(outBiasIndex, weightLen);
      const outputScratch = getMlpOutputScratch();
      for(let i = 0; i < prevSize; i++){
        const value = activations[i];
        if(!Number.isFinite(value) || value === 0){
          continue;
        }
        const wIdx = outWeightsBase + i;
        if(wIdx >= outLimit){
          break;
        }
        outputScratch[0] += value * weights[wIdx];
      }
      const bias = (outBiasIndex < weightLen) ? weights[outBiasIndex] : 0;
      return outputScratch[0] + bias;
    }
    function scoreFeats(weights, feats){ return isMlpModelType(train.modelType) ? mlpScore(weights, feats, train.modelType) : dot(weights, feats); }

    function evaluateAlphaPlacements(grid, curShape){
      return trainingProfiler.section('train.alpha.evaluate_all', () => {
        const placements = enumeratePlacements(grid, curShape);
        if(!placements.length){
          return [];
        }
        const model = ensureAlphaModelInstance();
        if(!model){
          return [];
        }

        const baselineMetrics = computeGridMetrics(grid);
        const baselineHoles = baselineMetrics ? baselineMetrics.holes : 0;
        for(let c = 0; c < WIDTH; c += 1){
          baselineColumnMaskScratch[c] = columnMaskScratch[c] || 0;
          baselineColumnHeightScratch[c] = columnHeightScratch[c] || 0;
        }
        const rootFeatureScratch = featuresFromGrid(grid, 0, {
          holeBaseline: baselineHoles,
          baselineColumnMasks: baselineColumnMaskScratch,
          clearedRows: null,
        });
        const rootEngineeredFeatures = new Float32Array(rootFeatureScratch.length);
        rootEngineeredFeatures.set(rootFeatureScratch);
        const alphaState = ensureAlphaState();
        const tf = (typeof window !== 'undefined' && window.tf) ? window.tf : null;
        if(!tf){
          if(typeof console !== 'undefined' && console.warn){
            console.warn('TensorFlow.js unavailable for AlphaTetris evaluation.');
          }
          return [];
        }

        let rootPolicyLogits = null;
        let rootMask = null;
        try {
          const baseInputs = prepareAlphaInputs({
            grid,
            active: state.active,
            next: state.next,
            preview: state.preview || null,
            nextQueue: state.nextQueue || null,
            score: state.score,
            level: state.level,
            pieces: state.pieces,
            gravity: state.gravity,
            lines: 0,
            newHoles: 0,
            engineeredFeatures: rootEngineeredFeatures,
          });
          rootMask = baseInputs.policyMask;
          if(alphaState){
            alphaState.lastPreparedInputs = {
              board: baseInputs.board,
              aux: baseInputs.aux,
              mask: baseInputs.policyMask ? new Float32Array(baseInputs.policyMask) : null,
            };
          }
          const rootResult = runAlphaInference(model, { board: baseInputs.board, aux: baseInputs.aux }, { reuseGraph: true, tf });
          if(rootResult && Array.isArray(rootResult.policyLogits) && rootResult.policyLogits.length){
            rootPolicyLogits = rootResult.policyLogits[0];
            alphaState.lastPolicyLogits = rootPolicyLogits ? rootPolicyLogits.slice() : null;
          } else {
            alphaState.lastPolicyLogits = null;
          }
          if(rootResult && rootResult.values && rootResult.values.length){
            const rv = rootResult.values[0];
            alphaState.lastRootValue = Number.isFinite(rv) ? rv : 0;
          } else {
            alphaState.lastRootValue = 0;
          }
        } catch (err) {
          if(typeof console !== 'undefined' && console.error){
            console.error('AlphaTetris root inference failed', err);
          }
          rootPolicyLogits = null;
          rootMask = null;
          alphaState.lastPolicyLogits = null;
          alphaState.lastRootValue = 0;
          if(alphaState){
            alphaState.lastPreparedInputs = null;
          }
        }

        const boards = [];
        const auxes = [];
        const candidates = [];
        const basePieces = Number.isFinite(state.pieces) ? state.pieces : 0;
        const baseLevel = Number.isFinite(state.level) ? state.level : 0;
        const baseScore = Number.isFinite(state.score) ? state.score : 0;
        const nextShape = state.next;

        for(const placement of placements){
          const policyIndex = ALPHA_ACTION_INDEX.get(`${placement.rot}|${placement.col}`);
          if(policyIndex === undefined){
            continue;
          }
          if(rootMask && rootMask.length > policyIndex && rootMask[policyIndex] <= 0){
            continue;
          }
          const sim = simulateAfterPlacement(grid, curShape, placement.rot, placement.col);
          if(!sim){
            continue;
          }
          const lines = Number.isFinite(sim.lines) ? sim.lines : 0;
          const dropRow = Number.isFinite(sim.dropRow) ? sim.dropRow : null;
          const newPieces = basePieces + 1;
          const newLevel = baseLevel + ((newPieces % 20 === 0) ? 1 : 0);
          const newGravity = gravityForLevel(newLevel);
          const scoreDelta = lines ? lines * 100 * (lines > 1 ? lines : 1) : 0;
          const newScore = baseScore + scoreDelta;
          const topOut = sim.grid[0].some((cell) => cell !== 0);

          let activePieceForNext = null;
          if(!topOut && nextShape){
            alphaSpawnPiece.shape = nextShape;
            alphaSpawnPiece.rot = 0;
            alphaSpawnPiece.row = 0;
            alphaSpawnPiece.col = Math.floor(WIDTH / 2) - 2;
            activePieceForNext = alphaSpawnPiece;
          }

          const featureScratch = featuresFromGrid(sim.grid, lines, {
            holeBaseline: baselineHoles,
            baselineColumnMasks: baselineColumnMaskScratch,
            clearedRows: sim.clearedRows,
          });
          const engineeredFeatures = new Float32Array(featureScratch.length);
          engineeredFeatures.set(featureScratch);
          const newHoleIndex = ENGINEERED_FEATURE_INDEX.NEW_HOLE_RATIO;
          const newHoleCount = engineeredFeatures.length > newHoleIndex
            ? Math.max(0, engineeredFeatures[newHoleIndex] * BOARD_AREA)
            : 0;
          const reward = computePlacementReward({
            lines,
            features: engineeredFeatures,
            baselineFeatures: rootEngineeredFeatures,
            newHoleCount,
            topOut,
          });

          const candidateState = {
            grid: sim.grid,
            active: activePieceForNext,
            next: null,
            score: newScore,
            level: newLevel,
            pieces: newPieces,
            gravity: newGravity,
            lines,
            newHoles: newHoleCount,
            engineeredFeatures,
          };

          const prepared = prepareAlphaInputs(candidateState);
          boards.push(prepared.board);
          auxes.push(prepared.aux);
          const clearedRowsCopy = sim.clearedRows && sim.clearedRows.length
            ? sim.clearedRows.slice()
            : [];
          candidates.push({
            key: `${placement.rot}|${placement.col}`,
            rot: placement.rot,
            col: placement.col,
            dropRow,
            lines,
            reward,
            policyIndex,
            topOut,
            policyLogit: 0,
            alpha: {
              value: 0,
              topOut,
              policyIndex,
              headless: {
                clearedRows: clearedRowsCopy,
                lines,
                nextPieces: newPieces,
                nextLevel: newLevel,
                nextGravity: newGravity,
                nextScore: newScore,
                topOut,
              },
            },
          });
        }

        const count = candidates.length;
        if(!count){
          if(alphaState){
            alphaState.lastPreparedInputs = null;
          }
          return [];
        }

        const boardTensorData = new Float32Array(count * ALPHA_BOARD_SIZE);
        const auxTensorData = new Float32Array(count * ALPHA_AUX_FEATURE_COUNT);
        for(let i = 0; i < count; i += 1){
          boardTensorData.set(boards[i], i * ALPHA_BOARD_SIZE);
          auxTensorData.set(auxes[i], i * ALPHA_AUX_FEATURE_COUNT);
        }

        let boardTensor = null;
        let auxTensor = null;
        let inferenceResult = null;
        try {
          boardTensor = tf.tensor(boardTensorData, [count, ALPHA_BOARD_HEIGHT, ALPHA_BOARD_WIDTH, ALPHA_BOARD_CHANNELS], 'float32');
          auxTensor = tf.tensor(auxTensorData, [count, ALPHA_AUX_FEATURE_COUNT], 'float32');
          inferenceResult = runAlphaInference(model, { boardTensor, auxTensor }, { reuseGraph: true, tf });
        } catch (err) {
          if(typeof console !== 'undefined' && console.error){
            console.error('AlphaTetris placement inference failed', err);
          }
          inferenceResult = null;
        } finally {
          if(boardTensor){ boardTensor.dispose(); }
          if(auxTensor){ auxTensor.dispose(); }
        }

        const values = inferenceResult && inferenceResult.values ? inferenceResult.values : null;
        for(let i = 0; i < candidates.length; i += 1){
          const candidate = candidates[i];
          const predicted = values && values.length > i ? values[i] : 0;
          const sanitized = Number.isFinite(predicted) ? predicted : 0;
          candidate.value = candidate.topOut ? ALPHA_TOP_OUT_VALUE : sanitized;
          candidate.alpha.value = sanitized;
          if(rootPolicyLogits && rootPolicyLogits.length > candidate.policyIndex){
            const logit = rootPolicyLogits[candidate.policyIndex];
            candidate.policyLogit = Number.isFinite(logit) ? logit : candidate.value;
          } else {
            candidate.policyLogit = candidate.value;
          }
        }

        return candidates;
      });
    }

    function evaluatePlacementsForSearch(weights, grid, curShape){
      return trainingProfiler.section('train.plan.evaluate_all', () => {
        if(isAlphaModelType(train.modelType)){
          const alphaEvaluations = evaluateAlphaPlacements(grid, curShape);
          if(alphaEvaluations && alphaEvaluations.length){
            return alphaEvaluations;
          }
        }
        const actions = [];
        const placements = enumeratePlacements(grid, curShape);
        if(placements.length === 0){
          return actions;
        }
        const baselineMetrics = computeGridMetrics(grid);
        const baselineHoles = baselineMetrics ? baselineMetrics.holes : 0;
        for(let c = 0; c < WIDTH; c += 1){
          baselineColumnMaskScratch[c] = columnMaskScratch[c] || 0;
          baselineColumnHeightScratch[c] = columnHeightScratch[c] || 0;
        }
        fillRowMaskFromGrid(grid, baselineRowMaskScratch);
        const baselineFeatureScratch = featuresFromGrid(grid, 0, {
          holeBaseline: baselineHoles,
          baselineColumnMasks: baselineColumnMaskScratch,
          clearedRows: null,
        });
        const baselineEngineeredFeatures = new Float32Array(baselineFeatureScratch.length);
        baselineEngineeredFeatures.set(baselineFeatureScratch);
        const needRawFeatures = isMlpModelType(train.modelType) && train.modelType === 'mlp_raw';
        for(const placement of placements){
          const key = placementSurrogateKeyFor(
            curShape,
            placement.rot,
            placement.col,
            baselineColumnMaskScratch,
            baselineColumnHeightScratch,
          );
          let surrogateResult = null;
          let attemptedSurrogate = false;
          if(placementSurrogate.enabled && key){
            attemptedSurrogate = true;
            placementSurrogate.stats.attempts += 1;
            const startEstimate = surrogateTimerNow();
            const estimate = estimatePlacementWithSurrogate(key, {
              baselineRowMasks: baselineRowMaskScratch,
              baselineColumnMasks: baselineColumnMaskScratch,
              baselineColumnHeights: baselineColumnHeightScratch,
              needRawFeatures,
            });
            placementSurrogate.stats.timeSurrogate += surrogateTimerNow() - startEstimate;
            if(
              estimate &&
              (!needRawFeatures || (estimate.rawFeatures && estimate.rawFeatures.length === RAW_FEAT_DIM))
            ){
              surrogateResult = estimate;
              placementSurrogate.stats.successes += 1;
            } else {
              placementSurrogate.stats.fallbacks += 1;
            }
          }
          let lines = 0;
          let dropRow = null;
          let topOut = false;
          let engineeredFeatures = null;
          let newHoleCount = 0;
          let baseFeats = null;
          let reward = 0;
          if(!surrogateResult){
            const simStart = attemptedSurrogate ? surrogateTimerNow() : null;
            const sim = simulateAfterPlacement(grid, curShape, placement.rot, placement.col);
            if(attemptedSurrogate && simStart !== null){
              placementSurrogate.stats.timeFallback += surrogateTimerNow() - simStart;
            }
            if(!sim){
              continue;
            }
            lines = Number.isFinite(sim.lines) ? sim.lines : 0;
            dropRow = sim.dropRow;
            topOut = sim.grid[0].some((cell) => cell !== 0);
            const engineeredScratch = featuresFromGrid(sim.grid, lines, {
              holeBaseline: baselineHoles,
              baselineColumnMasks: baselineColumnMaskScratch,
              clearedRows: sim.clearedRows,
            });
            engineeredFeatures = new Float32Array(engineeredScratch.length);
            engineeredFeatures.set(engineeredScratch);
            const newHoleIndex = ENGINEERED_FEATURE_INDEX.NEW_HOLE_RATIO;
            newHoleCount = engineeredFeatures.length > newHoleIndex
              ? Math.max(0, engineeredFeatures[newHoleIndex] * BOARD_AREA)
              : 0;
            reward = computePlacementReward({
              lines,
              features: engineeredFeatures,
              baselineFeatures: baselineEngineeredFeatures,
              newHoleCount,
              topOut,
            });
            baseFeats = needRawFeatures ? rawFeaturesFromGrid(sim.grid) : engineeredFeatures;
          } else {
            lines = Number.isFinite(surrogateResult.lines) ? surrogateResult.lines : 0;
            dropRow = surrogateResult.dropRow;
            topOut = !!surrogateResult.topOut;
            engineeredFeatures = surrogateResult.engineeredFeatures;
            const newHoleIndex = ENGINEERED_FEATURE_INDEX.NEW_HOLE_RATIO;
            newHoleCount = engineeredFeatures.length > newHoleIndex
              ? Math.max(0, engineeredFeatures[newHoleIndex] * BOARD_AREA)
              : 0;
            reward = computePlacementReward({
              lines,
              features: engineeredFeatures,
              baselineFeatures: baselineEngineeredFeatures,
              newHoleCount,
              topOut,
            });
            baseFeats = needRawFeatures ? surrogateResult.rawFeatures : engineeredFeatures;
            if(placementSurrogate.debugCompare){
              const debugStart = surrogateTimerNow();
              const sim = simulateAfterPlacement(grid, curShape, placement.rot, placement.col);
              placementSurrogate.stats.timeDebugCompare += surrogateTimerNow() - debugStart;
              if(sim){
                placementSurrogate.stats.comparisons += 1;
                const simLines = Number.isFinite(sim.lines) ? sim.lines : 0;
                const simTopOut = sim.grid[0].some((cell) => cell !== 0);
                const simFeaturesScratch = featuresFromGrid(sim.grid, simLines, {
                  holeBaseline: baselineHoles,
                  baselineColumnMasks: baselineColumnMaskScratch,
                  clearedRows: sim.clearedRows,
                });
                const simFeatures = new Float32Array(simFeaturesScratch.length);
                simFeatures.set(simFeaturesScratch);
                let mismatch = false;
                if(simLines !== lines || simTopOut !== topOut){
                  mismatch = true;
                } else {
                  const tol = 1e-4;
                  const len = Math.min(simFeatures.length, engineeredFeatures.length);
                  for(let i = 0; i < len; i += 1){
                    if(Math.abs(simFeatures[i] - engineeredFeatures[i]) > tol){
                      mismatch = true;
                      break;
                    }
                  }
                }
                if(mismatch){
                  placementSurrogate.stats.mismatches += 1;
                  if(typeof console !== 'undefined' && console.warn){
                    console.warn('Surrogate mismatch', {
                      shape: curShape,
                      rot: placement.rot,
                      col: placement.col,
                      surrogate: { lines, topOut, features: engineeredFeatures },
                      exact: { lines: simLines, topOut: simTopOut, features: simFeatures },
                    });
                  }
                }
              }
            }
          }
          const value = scoreFeats(weights, baseFeats);
          actions.push({
            key: key || `${placement.rot}|${placement.col}`,
            rot: placement.rot,
            col: placement.col,
            dropRow,
            lines,
            value: Number.isFinite(value) ? value : 0,
            reward: Number.isFinite(reward) ? reward : 0,
          });
        }
        maybeLogPlacementSurrogateStats();
        return actions;
      });
    }

    function choosePlacement(weights, grid, curShape){
      return trainingProfiler.section('train.plan.single', () => {
        const evaluations = evaluatePlacementsForSearch(weights, grid, curShape);
        if(evaluations.length === 0){
          return null;
        }
        if(isAlphaModelType(train.modelType)){
          let bestEntry = null;
          for(let i = 0; i < evaluations.length; i += 1){
            const candidate = evaluations[i];
            const metric = Number.isFinite(candidate.policyLogit) ? candidate.policyLogit : candidate.value;
            if(!bestEntry || metric > bestEntry.metric){
              bestEntry = { candidate, metric };
            }
          }
          const selected = bestEntry ? bestEntry.candidate : evaluations[0];
          return { rot: selected.rot, col: selected.col, dropRow: selected.dropRow, lines: selected.lines };
        }
        let best = evaluations[0];
        for(let i = 1; i < evaluations.length; i += 1){
          const candidate = evaluations[i];
          if(candidate.value > best.value){
            best = candidate;
          }
        }
        return { rot: best.rot, col: best.col, dropRow: best.dropRow, lines: best.lines };
      });
    }

    function planForCurrentPiece(){
      return trainingProfiler.section('train.plan', () => {
        if(!state.active){
          return null;
        }
        const w = train.currentWeightsOverride || train.candWeights[train.candIndex] || train.mean;
        if(usesPopulationModel(train.modelType)){
          const placement = choosePlacement(w, state.grid, state.active.shape);
          if(!placement){
            train.ai.lastSearchStats = null;
            return null;
          }
          const len = SHAPES[state.active.shape].length;
          const cur = state.active.rot % len;
          const needRot = (placement.rot - cur + len) % len;
          const targetRow = Number.isFinite(placement.dropRow) ? placement.dropRow : null;
          train.ai.lastSearchStats = null;
          return {
            targetRot: placement.rot,
            targetCol: placement.col,
            targetRow,
            rotLeft: needRot,
            stage: 'rotate',
            search: null,
          };
        }
        const evaluations = evaluatePlacementsForSearch(w, state.grid, state.active.shape);
        if(!evaluations.length){
          train.ai.lastSearchStats = null;
          return null;
        }

        const usePolicyLogits = isAlphaModelType(train.modelType);
        const logits = evaluations.map((action) => {
          if(usePolicyLogits){
            return Number.isFinite(action.policyLogit) ? action.policyLogit : action.value;
          }
          return action.value;
        });
        const priors = softmax(logits);
        for(let i = 0; i < evaluations.length; i += 1){
          evaluations[i].prior = priors.length > i ? priors[i] : 0;
        }
        normalizePriors(evaluations);

        if(!train.ai.search){
          train.ai.search = { simulations: 48, cPuct: 1.5, temperature: 1, discount: 1 };
        }
        const searchConfig = train.ai.search;
        const simulationTarget = sanitizeSimulationCount(searchConfig.simulations, searchConfig.simulations);
        searchConfig.simulations = simulationTarget;
        const cPuct = sanitizeExplorationConstant(searchConfig.cPuct, searchConfig.cPuct);
        searchConfig.cPuct = cPuct;
        const temperature = sanitizeTemperature(searchConfig.temperature, searchConfig.temperature);
        searchConfig.temperature = temperature;
        const discount = Number.isFinite(searchConfig.discount) ? searchConfig.discount : 1;

        const root = createNode({ expanded: true });
        for(const action of evaluations){
          const key = action.key;
          const child = createNode({
            parent: root,
            prior: action.prior,
            reward: action.reward,
            valueEstimate: action.value,
            metadata: action,
          });
          root.children.set(key, child);
        }

        const simulationLimit = Math.max(1, Math.min(simulationTarget, MAX_AI_STEPS_PER_FRAME));
        for(let sim = 0; sim < simulationLimit; sim += 1){
          const path = [{ node: root, reward: 0 }];
          let node = root;
          while(node.children && node.children.size > 0){
            const [, child] = selectChild(node, cPuct, discount);
            if(!child){
              break;
            }
            path.push({ node: child, reward: child.reward });
            node = child;
            if(!child.expanded || !child.children || child.children.size === 0){
              break;
            }
          }
          const leafValue = Number.isFinite(node.valueEstimate) ? node.valueEstimate : 0;
          node.expanded = true;
          backpropagate(path, leafValue, discount);
        }

        const stats = visitStats(root, discount);
        const policyDistribution = computeVisitPolicy(stats, temperature);
        let chosen = null;
        if(temperature === 0){
          chosen = policyDistribution.find((entry) => entry.policy === 1) || null;
        }
        if(!chosen){
          chosen = sampleFromPolicy(policyDistribution) || null;
        }
        if(!chosen && policyDistribution.length){
          chosen = policyDistribution[0];
        }

        const selected = chosen || null;
        const selectedMeta = selected && selected.metadata ? selected.metadata : null;
        const totalVisits = stats.reduce((acc, entry) => acc + (entry.visits || 0), 0);
        const policyStats = policyDistribution.map((entry) => ({
          key: entry.key,
          visits: entry.visits,
          prior: entry.prior,
          value: entry.value,
          reward: entry.reward,
          policy: entry.policy,
          rot: entry.metadata ? entry.metadata.rot : null,
          col: entry.metadata ? entry.metadata.col : null,
          dropRow: entry.metadata ? entry.metadata.dropRow : null,
          lines: entry.metadata ? entry.metadata.lines : null,
        }));
        const alphaRootValue = usePolicyLogits && train.alpha && Number.isFinite(train.alpha.lastRootValue)
          ? train.alpha.lastRootValue
          : null;
        const avgRootValue = root.visitCount > 0 ? root.valueSum / root.visitCount : NaN;
        const selectedValue = selectedMeta && Number.isFinite(selectedMeta.value)
          ? selectedMeta.value
          : 0;
        const rootValue = Number.isFinite(avgRootValue)
          ? avgRootValue
          : (alphaRootValue !== null ? alphaRootValue : selectedValue);
        if(usePolicyLogits){
          recordAlphaTrainingExample(policyStats, rootValue);
        }
        if(!selectedMeta){
          train.ai.lastSearchStats = {
            rootValue,
            totalVisits,
            simulations: simulationLimit,
            temperature,
            cPuct,
            policy: policyStats,
            selected: null,
          };
          return null;
        }

        const len = SHAPES[state.active.shape].length;
        const cur = state.active.rot % len;
        const needRot = (selectedMeta.rot - cur + len) % len;
        const targetRow = Number.isFinite(selectedMeta.dropRow) ? selectedMeta.dropRow : null;
        const rawHeadless = selectedMeta.alpha && selectedMeta.alpha.headless
          ? selectedMeta.alpha.headless
          : null;
        const alphaHeadless = rawHeadless
          ? {
              clearedRows: Array.isArray(rawHeadless.clearedRows)
                ? rawHeadless.clearedRows.slice()
                : [],
              lines: rawHeadless.lines,
              nextPieces: rawHeadless.nextPieces,
              nextLevel: rawHeadless.nextLevel,
              nextGravity: rawHeadless.nextGravity,
              nextScore: rawHeadless.nextScore,
              topOut: !!rawHeadless.topOut,
            }
          : null;
        train.ai.lastSearchStats = {
          rootValue,
          totalVisits,
          simulations: simulationLimit,
          temperature,
          cPuct,
          policy: policyStats,
          selected: {
            key: selected.key,
            rot: selectedMeta.rot,
            col: selectedMeta.col,
            dropRow: selectedMeta.dropRow,
            lines: selectedMeta.lines,
            value: selected.value,
            reward: selected.reward,
            visits: selected.visits,
            prior: selected.prior,
            policy: selected.policy,
          },
        };

        return {
          targetRot: selectedMeta.rot,
          targetCol: selectedMeta.col,
          targetRow,
          rotLeft: needRot,
          stage: 'rotate',
          search: {
            totalVisits,
            simulations: simulationLimit,
            temperature,
            cPuct,
            policy: policyStats,
            selected: train.ai.lastSearchStats.selected,
          },
          alphaHeadless,
        };
      });
    }

    // Scatter plot of score history. When plotBestOnly is true, only best-of-generation points are rendered.
    function updateScorePlot(){
      const canvas = document.getElementById('score-plot');
      if(!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0,0,W,H);

      const padL = 48;
      const padR = 24;
      const padT = 26;
      const padB = 44;
      const axisColor = 'rgba(249, 245, 255, 0.68)';
      const gridColor = 'rgba(249, 245, 255, 0.1)';

      const trainState = window.__train || null;
      const usingPopulation = trainState ? usesPopulationModel(trainState.modelType) : true;
      const usingBestSnapshots = !!(trainState && trainState.plotBestOnly);
      const xw = Math.max(0, W - padL - padR);
      const yh = Math.max(0, H - padT - padB);

      const dataset = [];
      let maxXValue = 0;
      let minXValue = Infinity;
      if(usingBestSnapshots){
        const snapshots = trainState && Array.isArray(trainState.bestByGeneration)
          ? trainState.bestByGeneration
          : [];
        for(let i = 0; i < snapshots.length; i += 1){
          const entry = snapshots[i];
          if(!entry) continue;
          const score = Number.isFinite(entry.fitness) ? entry.fitness : 0;
          const type = entry.modelType || (trainState && trainState.modelType) || 'linear';
          const rawX = Number.isFinite(entry.gen) ? entry.gen : i + 1;
          const xValue = rawX > 0 ? rawX : i + 1;
          dataset.push({ score, type, xValue });
          if(xValue > maxXValue){
            maxXValue = xValue;
          }
          if(xValue < minXValue){
            minXValue = xValue;
          }
        }
      } else {
        const scores = trainState && Array.isArray(trainState.gameScores) ? trainState.gameScores : [];
        const types  = trainState && Array.isArray(trainState.gameModelTypes) ? trainState.gameModelTypes : [];
        const offset = Number.isFinite(trainState && trainState.gameScoresOffset) ? trainState.gameScoresOffset : 0;
        for(let i = 0; i < scores.length; i += 1){
          const rawScore = scores[i];
          const score = Number.isFinite(rawScore) ? rawScore : 0;
          const type = types[i] || 'linear';
          const xValue = offset + i + 1;
          dataset.push({ score, type, xValue, scoreIndex: offset + i });
          if(xValue > maxXValue){
            maxXValue = xValue;
          }
          if(xValue < minXValue){
            minXValue = xValue;
          }
        }
      }

      const count = dataset.length;
      if(trainState && typeof trainState.scorePlotPending !== 'number'){
        trainState.scorePlotPending = 0;
      }
      if(!count){
        if(trainState){
          trainState.scorePlotPending = 0;
          if(!Number.isFinite(trainState.scorePlotAxisMax) || trainState.scorePlotAxisMax < 1){
            const genCount = Number.isFinite(trainState.gen) ? trainState.gen : 0;
            const popBaseline = Math.max(1, Number.isFinite(trainState.popSize) ? trainState.popSize : 0);
            const gameBaseline = Math.max(10, Number.isFinite(trainState.totalGamesPlayed) ? trainState.totalGamesPlayed : 0);
            const baselineSource = usingBestSnapshots
              ? Math.max(1, genCount, usingPopulation ? 1 : 10)
              : (usingPopulation ? popBaseline : gameBaseline);
            const baseline = Math.max(10, Math.ceil(baselineSource * 1.2));
            const maxCap = Math.max(1, trainState.maxPlotPoints || baseline);
            trainState.scorePlotAxisMax = Math.min(maxCap, baseline);
          }
        }
        return;
      }

      maxXValue = Math.max(maxXValue, count);
      const maxScore = dataset.reduce((acc, point) => (point.score > acc ? point.score : acc), 0);
      let maxY = Math.ceil(Math.max(10000, maxScore) / 10000) * 10000;
      if(!Number.isFinite(maxY) || maxY <= 0){
        maxY = 10000;
      }

      const yTicks = [];
      for(let tick = 0; tick <= maxY; tick += 10000){
        yTicks.push(tick);
      }
      if(yTicks[yTicks.length - 1] !== maxY){
        yTicks.push(maxY);
      }

      ctx.lineWidth = 1;
      ctx.strokeStyle = gridColor;
      yTicks.forEach((tick) => {
        const y = H - padB - (tick / maxY) * yh;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
      });

      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, H - padB);
      ctx.lineTo(W - padR, H - padB);
      ctx.stroke();

      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1;
      ctx.fillStyle = axisColor;
      ctx.font = '11px "Instrument Serif", serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      yTicks.forEach((tick) => {
        const y = H - padB - (tick / maxY) * yh;
        ctx.beginPath();
        ctx.moveTo(padL - 6, y);
        ctx.lineTo(padL, y);
        ctx.stroke();
        ctx.fillText(tick.toLocaleString(), padL - 10, y);
      });

      let axisMax = maxXValue;
      if(trainState){
        const maxCap = Math.max(maxXValue, trainState.maxPlotPoints || maxXValue);
        let currentAxis = Number.isFinite(trainState.scorePlotAxisMax) ? trainState.scorePlotAxisMax : 0;
        if(currentAxis < 1){
          const genCount = Number.isFinite(trainState.gen) ? trainState.gen : 0;
          const popBaseline = Math.max(1, Number.isFinite(trainState.popSize) ? trainState.popSize : 0);
          const gameBaseline = Math.max(10, Number.isFinite(trainState.totalGamesPlayed) ? trainState.totalGamesPlayed : 0);
          const fallbackCount = usingPopulation
            ? Math.max(1, popBaseline, count || 0, 5)
            : Math.max(gameBaseline, count || 0);
          const baselineSource = usingBestSnapshots
            ? Math.max(maxXValue, count, genCount, usingPopulation ? 5 : 10)
            : fallbackCount;
          const baseline = Math.max(10, Math.ceil(baselineSource * 1.2));
          currentAxis = Math.min(maxCap, baseline);
        }
        if(maxXValue > currentAxis){
          let next = Math.ceil(currentAxis * 1.2);
          if(!Number.isFinite(next) || next <= currentAxis){
            next = currentAxis + 1;
          }
          currentAxis = Math.min(maxCap, Math.max(next, maxXValue));
        }
        trainState.scorePlotAxisMax = currentAxis;
        trainState.scorePlotPending = 0;
        axisMax = Math.max(maxXValue, currentAxis);
      }

      const denom = axisMax > 1 ? axisMax - 1 : 1;
      const desiredTicks = Math.min(8, Math.max(3, Math.round(xw / 70)));
      let step = 1;
      if(axisMax > 1){
        const raw = denom / Math.max(1, desiredTicks - 1);
        const exponent = Math.floor(Math.log10(raw));
        const base = Math.pow(10, exponent);
        const fraction = raw / base;
        let niceFraction;
        if(fraction >= 5){
          niceFraction = 5;
        } else if(fraction >= 2){
          niceFraction = 2;
        } else {
          niceFraction = 1;
        }
        step = Math.max(1, Math.round(niceFraction * base));
      }

      const tickSet = new Set();
      for(let tick = 1; tick <= axisMax; tick += step){
        tickSet.add(Math.round(tick));
      }
      tickSet.add(axisMax);
      tickSet.add(Math.round(maxXValue));
      const xTicks = Array.from(tickSet)
        .filter((tick) => tick >= 1 && tick <= axisMax)
        .sort((a,b) => a - b);

      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = axisColor;
      ctx.strokeStyle = axisColor;
      ctx.lineWidth = 1;
      xTicks.forEach((tick) => {
        const ratio = axisMax <= 1 ? 1 : (tick - 1) / denom;
        const x = padL + ratio * xw;
        ctx.beginPath();
        ctx.moveTo(x, H - padB);
        ctx.lineTo(x, H - padB + 6);
        ctx.stroke();
        ctx.fillText(String(tick), x, H - padB + 8);
      });

      const COLORS = { linear: '#76b3ff', mlp: '#ff9a6b', mlp_raw: '#facc15' };
      const safeMaxY = maxY || 1;
      const pointPositions = [];
      for(let i = 0; i < dataset.length; i += 1){
        const point = dataset[i];
        const xVal = Math.max(1, Number.isFinite(point.xValue) ? point.xValue : i + 1);
        const ratio = axisMax <= 1 ? 1 : (xVal - 1) / denom;
        const x = padL + ratio * xw;
        const y = H - padB - (point.score / safeMaxY) * yh;
        const color = COLORS[point.type || 'linear'] || COLORS.linear;
        pointPositions.push({ x, y });
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = 'rgba(12, 17, 32, 0.85)';
        ctx.stroke();
      }

      const firstPlottedX = Number.isFinite(minXValue) ? Math.max(1, minXValue) : 1;
      const computePlotX = (value, fallbackIndex) => {
        let base = Number.isFinite(value) ? value : null;
        if(base === null || base <= 1){
          base = fallbackIndex + 1;
        }
        const clamped = Math.min(axisMax, Math.max(1, base));
        const ratio = axisMax <= 1 ? 1 : (clamped - 1) / denom;
        return padL + ratio * xw;
      };

      // Draw a centered rolling average (last 5 and next 5 points). Prefer best-of-generation
      // snapshots when available so the trend line matches the data being emphasized.
      const windowRadius = 5;
      const requiredWindow = windowRadius * 2 + 1;
      const bestSeries = trainState && Array.isArray(trainState.bestByGeneration)
        ? trainState.bestByGeneration
        : null;
      const rollingSeries = (() => {
        if(bestSeries && bestSeries.length){
          const series = [];
          for(let i = 0; i < bestSeries.length; i += 1){
            const entry = bestSeries[i];
            if(!entry) continue;
            const score = Number.isFinite(entry.fitness) ? entry.fitness : 0;
            let xRaw;
            if(usingBestSnapshots){
              xRaw = Number.isFinite(entry.gen) && entry.gen > 0 ? entry.gen : i + 1;
            } else if(Number.isFinite(entry.scoreIndex)){
              xRaw = entry.scoreIndex + 1;
            } else if(Number.isFinite(entry.gen) && entry.gen > 0){
              xRaw = entry.gen;
            } else {
              xRaw = i + 1;
            }
            if(!usingBestSnapshots && xRaw < firstPlottedX){
              continue;
            }
            series.push({
              score,
              x: computePlotX(xRaw, i),
            });
          }
          if(series.length >= requiredWindow){
            return series;
          }
        }
        return dataset.map((point, idx) => {
          const plotPoint = pointPositions[idx];
          const score = Number.isFinite(point.score) ? point.score : 0;
          const x = plotPoint && Number.isFinite(plotPoint.x)
            ? plotPoint.x
            : computePlotX(point.xValue, idx);
          return { score, x };
        });
      })();

      if(rollingSeries.length >= requiredWindow){
        const prefix = new Array(rollingSeries.length + 1);
        prefix[0] = 0;
        for(let i = 0; i < rollingSeries.length; i += 1){
          prefix[i + 1] = prefix[i] + rollingSeries[i].score;
        }
        const startIdx = windowRadius;
        const endIdx = rollingSeries.length - windowRadius - 1;
        if(startIdx <= endIdx){
          ctx.save();
          ctx.beginPath();
          ctx.lineWidth = 1;
          ctx.strokeStyle = '#ffffff';
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          let drewPoint = false;
          for(let i = startIdx; i <= endIdx; i += 1){
            const windowStart = i - windowRadius;
            const windowEnd = i + windowRadius;
            const windowCount = windowEnd - windowStart + 1;
            if(windowCount <= 0) continue;
            const sum = prefix[windowEnd + 1] - prefix[windowStart];
            const avg = sum / windowCount;
            const x = rollingSeries[i] && Number.isFinite(rollingSeries[i].x)
              ? rollingSeries[i].x
              : computePlotX(null, i);
            if(!Number.isFinite(x)) continue;
            const y = H - padB - (avg / safeMaxY) * yh;
            if(!drewPoint){
              ctx.moveTo(x, y);
              drewPoint = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
          if(drewPoint){
            ctx.stroke();
          }
          ctx.restore();
        }
      }

      if(trainState && Array.isArray(trainState.bestByGeneration) && trainState.bestByGeneration.length){
        let selection = trainState.historySelection;
        if(selection !== null && selection !== undefined){
          selection = Math.max(0, Math.min(trainState.bestByGeneration.length - 1, Math.round(selection)));
          if(usingBestSnapshots){
            const point = pointPositions[selection];
            if(point && Number.isFinite(point.x) && Number.isFinite(point.y)){
              ctx.save();
              ctx.beginPath();
              ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
              ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
              ctx.fill();
              ctx.beginPath();
              ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
              ctx.fillStyle = '#3b82f6';
              ctx.fill();
              ctx.lineWidth = 2;
              ctx.strokeStyle = '#1d4ed8';
              ctx.stroke();
              ctx.beginPath();
              ctx.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
              ctx.fillStyle = '#bfdbfe';
              ctx.fill();
              ctx.restore();
            }
          } else {
            const offset = Number.isFinite(trainState.gameScoresOffset) ? trainState.gameScoresOffset : 0;
            const snapshot = trainState.bestByGeneration[selection];
            const hasIndex = snapshot && Number.isFinite(snapshot.scoreIndex);
            if(hasIndex){
              const relative = Math.round(snapshot.scoreIndex - offset);
              if(relative >= 0 && relative < pointPositions.length){
                const point = pointPositions[relative];
                if(point && Number.isFinite(point.x) && Number.isFinite(point.y)){
                  ctx.save();
                  ctx.beginPath();
                  ctx.arc(point.x, point.y, 7, 0, Math.PI * 2);
                  ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
                  ctx.fill();
                  ctx.beginPath();
                  ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
                  ctx.fillStyle = '#3b82f6';
                  ctx.fill();
                  ctx.lineWidth = 2;
                  ctx.strokeStyle = '#1d4ed8';
                  ctx.stroke();
                  ctx.beginPath();
                  ctx.arc(point.x, point.y, 2.5, 0, Math.PI * 2);
                  ctx.fillStyle = '#bfdbfe';
                  ctx.fill();
                  ctx.restore();
                }
              }
            }
          }
        }
      }
    }

    function applyAlphaHeadlessPlacement(plan, alphaData){
      if(!plan || !alphaData || !state.active){
        return null;
      }
      const { targetRot, targetCol, targetRow } = plan;
      if(!Number.isFinite(targetRot) || !Number.isFinite(targetCol) || !Number.isFinite(targetRow)){
        return null;
      }
      const piece = state.active;
      const prevRot = piece.rot;
      const prevCol = piece.col;
      const prevRow = piece.row;

      piece.rot = targetRot;
      piece.col = targetCol;
      piece.row = targetRow;

      if(!canMove(state.grid, piece, 0, 0)){
        piece.rot = prevRot;
        piece.col = prevCol;
        piece.row = prevRow;
        return null;
      }

      state.active = piece;
      lock(state.grid, piece);

      let clearedCount = 0;
      if(Array.isArray(alphaData.clearedRows) && alphaData.clearedRows.length){
        clearedCount = applyClearedRowsToGrid(state.grid, alphaData.clearedRows);
      }

      const prevPieces = Number.isFinite(state.pieces) ? state.pieces : 0;
      const nextPieces = Number.isFinite(alphaData.nextPieces) ? alphaData.nextPieces : prevPieces + 1;
      state.pieces = nextPieces;

      const prevLevel = Number.isFinite(state.level) ? state.level : 0;
      const fallbackLevel = prevLevel + (nextPieces % 20 === 0 ? 1 : 0);
      const nextLevel = Number.isFinite(alphaData.nextLevel) ? alphaData.nextLevel : fallbackLevel;
      const levelChanged = nextLevel !== prevLevel;
      const nextGravity = Number.isFinite(alphaData.nextGravity) ? alphaData.nextGravity : gravityForLevel(nextLevel);
      state.level = nextLevel;
      state.gravity = nextGravity;
      if(levelChanged){
        updateLevel();
      }

      const prevScore = Number.isFinite(state.score) ? state.score : 0;
      const lines = Number.isFinite(alphaData.lines) ? alphaData.lines : clearedCount;
      const fallbackScoreDelta = lines ? lines * 100 * (lines > 1 ? lines : 1) : 0;
      const nextScore = Number.isFinite(alphaData.nextScore) ? alphaData.nextScore : prevScore + fallbackScoreDelta;
      state.score = nextScore;
      if(lines > 0){
        updateScore();
        recordClear(lines);
      }

      if(maybeEndForLevelCap('AI')){
        return false;
      }

      if(alphaData.topOut){
        logTrainingEvent('AI: top-out after drop');
        resetAiPlanState();
        onGameOver();
        return false;
      }

      spawn();
      resetAiPlanState();
      if(!canMove(state.grid, state.active, 0, 0)) onGameOver();
      return false;
    }

    function runHeadlessPlacement(){
      return trainingProfiler.section('train.ai.headless_placement', () => {
        if(!state.active){
          return false;
        }

        const finalizePlacement = (piece, topOutLog) => {
          state.active = piece;
          lock(state.grid, piece);
          state.pieces++;
          if(state.pieces % 20 === 0){
            state.level++;
            state.gravity = gravityForLevel(state.level);
            updateLevel();
          }
          const cleared = clearRows(state.grid);
          if(cleared){
            state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
            updateScore();
            recordClear(cleared);
          }
          if(maybeEndForLevelCap('AI')){
            return false;
          }
          const topOut = state.grid[0].some((v) => v !== 0);
          if(topOut){
            if(topOutLog){
              logTrainingEvent(topOutLog);
            }
            resetAiPlanState();
            onGameOver();
            return false;
          }
          spawn();
          resetAiPlanState();
          if(!canMove(state.grid, state.active, 0, 0)) onGameOver();
          return false;
        };

        const forceDropActive = () => {
          while(canMove(state.grid, state.active, 0, 1)){
            state.active.move(0, 1);
          }
          return finalizePlacement(state.active, 'AI: top-out after forced drop');
        };

        const plan = planForCurrentPiece();
        train.ai.plan = plan || null;
        if(!plan){
          return forceDropActive();
        }

        const { targetRot, targetCol, targetRow } = plan;
        if(!Number.isFinite(targetRot) || !Number.isFinite(targetCol) || !Number.isFinite(targetRow)){
          return forceDropActive();
        }

        if(isAlphaModelType(train.modelType) && plan.alphaHeadless){
          const alphaResult = applyAlphaHeadlessPlacement(plan, plan.alphaHeadless);
          if(alphaResult !== null){
            return alphaResult;
          }
        }

        const piece = state.active;
        const prevRot = piece.rot;
        const prevCol = piece.col;
        const prevRow = piece.row;

        piece.rot = targetRot;
        piece.col = targetCol;
        piece.row = targetRow;

        const fits = canMove(state.grid, piece, 0, 0);
        const settled = fits && !canMove(state.grid, piece, 0, 1);
        if(!fits || !settled){
          piece.rot = prevRot;
          piece.col = prevCol;
          piece.row = prevRow;
          return forceDropActive();
        }

        return finalizePlacement(piece, 'AI: top-out after drop');
      });
    }

    function runAiMicroStep(){
      return trainingProfiler.section('train.ai.micro_step', () => {
        if(!state.active){
          return false;
        }

        // Watchdog: if the active piece hasn't changed state for a while, force a drop
        const sig = `${state.active.shape}:${state.active.rot}:${state.active.row}:${state.active.col}:${state.score}`;
        if(train.ai.lastSig === sig){
          train.ai.staleMs = (train.ai.staleMs || 0) + AI_STEP_MS;
        } else {
          train.ai.staleMs = 0;
          train.ai.lastSig = sig;
        }
        if(train.ai.staleMs > 1000){
          logTrainingEvent('AI: watchdog forced drop');
          while(canMove(state.grid, state.active, 0, 1)) state.active.move(0,1);
          lock(state.grid, state.active);
          state.pieces++;
          if(state.pieces % 20 === 0){
            state.level++;
            state.gravity = gravityForLevel(state.level);
            updateLevel();
          }
          const cleared = clearRows(state.grid);
          if(cleared){
            state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
            updateScore();
            recordClear(cleared);
          }
          if(maybeEndForLevelCap('AI')){
            return false;
          }
          if(state.grid[0].some((v) => v !== 0)) {
            resetAiPlanState();
            onGameOver();
            return false;
          }
          spawn();
          resetAiPlanState();
          if(!canMove(state.grid, state.active, 0, 0)) onGameOver();
          return false;
        }

        if(!train.ai.plan){
          train.ai.plan = planForCurrentPiece();
          if(!train.ai.plan){
            // force drop to end episode
            while(canMove(state.grid, state.active, 0, 1)) state.active.move(0,1);
            lock(state.grid, state.active);
            state.pieces++;
            if(state.pieces % 20 === 0){
              state.level++;
              state.gravity = gravityForLevel(state.level);
              updateLevel();
            }
            const cleared = clearRows(state.grid);
            if(cleared){
              state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
              updateScore();
              recordClear(cleared);
            }
            if(maybeEndForLevelCap('AI')){
              return false;
            }
            if(state.grid[0].some((v) => v !== 0)) {
              logTrainingEvent('AI: top-out after forced drop');
              resetAiPlanState();
              onGameOver();
              return false;
            }
            spawn();
            resetAiPlanState();
            if(!canMove(state.grid, state.active, 0, 0)) onGameOver();
            return false;
          }
        }

        const plan = train.ai.plan;
        if(!plan) return true;
        if(plan.stage === 'rotate'){
          if(plan.rotLeft > 0){
            state.active.rotate();
            if(!canMove(state.grid, state.active, 0, 0)){
              // Rotation blocked: abandon this plan to avoid stalling
              state.active.rotate(-1);
              train.ai.plan = null;
              logTrainingEvent('AI: rotation blocked, abandoning plan');
            } else {
              plan.rotLeft -= 1;
            }
            return true;
          }
          plan.stage = 'move';
          return true;
        }
        if(plan.stage === 'move'){
          if(state.active.col < plan.targetCol){
            if(canMove(state.grid, state.active, 1, 0)){
              state.active.move(1,0);
            } else {
              plan.stage = 'drop';
            }
            return true;
          }
          if(state.active.col > plan.targetCol){
            if(canMove(state.grid, state.active, -1, 0)){
              state.active.move(-1,0);
            } else {
              plan.stage = 'drop';
            }
            return true;
          }
          plan.stage = 'drop';
          return true;
        }
        if(plan.stage === 'drop'){
          if(canMove(state.grid, state.active, 0, 1)){
            state.active.move(0,1);
            return true;
          }
          lock(state.grid, state.active);
          state.pieces++;
          if(state.pieces % 20 === 0){
            state.level++;
            state.gravity = gravityForLevel(state.level);
            updateLevel();
          }
          const cleared = clearRows(state.grid);
          if(cleared){
            state.score += cleared * 100 * (cleared > 1 ? cleared : 1);
            updateScore();
            recordClear(cleared);
          }
          if(maybeEndForLevelCap('AI')){
            return false;
          }
          if(state.grid[0].some((v) => v !== 0)) {
            logTrainingEvent('AI: top-out after drop');
            resetAiPlanState();
            onGameOver();
            return false;
          }
          spawn();
          resetAiPlanState();
          if(!canMove(state.grid, state.active, 0, 0)) onGameOver();
          return false;
        }
        return true;
      });
    }

    function aiStep(dt){
      if(!state.active){ return; }
      if(!canMove(state.grid, state.active, 0, 0)) { logTrainingEvent('AI: spawn blocked -> game over'); onGameOver(); return; }

      const headlessTraining = train && train.enabled && train.visualizeBoard === false;
      if(headlessTraining){
        train.ai.acc = 0;
        runHeadlessPlacement();
        return;
      }

      let __aiSteps = 0;
      train.ai.acc += dt;
      while (train.ai.acc >= AI_STEP_MS && __aiSteps < MAX_AI_STEPS_PER_FRAME) {
        train.ai.acc -= AI_STEP_MS;
        __aiSteps += 1;
        if(!runAiMicroStep()){
          return;
        }
      }
    }
    window.__aiStep = aiStep;

    const downloadBtn = document.getElementById('download-weights');
    if(downloadBtn){
      downloadBtn.addEventListener('click', () => { void downloadCurrentWeights(); });
    }
    const uploadInput = document.getElementById('upload-weights');
    const uploadBtn = document.getElementById('upload-weights-button');
    if(uploadBtn && uploadInput){
      uploadBtn.addEventListener('click', () => {
        uploadInput.click();
      });
      uploadInput.addEventListener('change', () => {
        const file = uploadInput.files && uploadInput.files[0];
        if(!file){
          return;
        }
        const reader = new FileReader();
        reader.onload = async (event) => {
          try {
            const text = typeof event.target.result === 'string' ? event.target.result : '';
            const snapshot = parseWeightSnapshot(text);
            await applyWeightSnapshot(snapshot, { fileName: file.name });
          } catch (err) {
            console.error(err);
            const message = (err && err.message) ? err.message : 'unknown error';
            log(`Failed to load weights: ${message}`);
          } finally {
            uploadInput.value = '';
          }
        };
        reader.onerror = () => {
          log('Failed to read weight file.');
          uploadInput.value = '';
        };
        reader.readAsText(file);
      });
    }

    const handleHistorySliderInput = () => {
      if(!historySlider || !train){
        return;
      }
      if(!Array.isArray(train.bestByGeneration) || train.bestByGeneration.length === 0){
        train.historySelection = null;
      } else {
        const raw = Number(historySlider.value);
        const total = train.bestByGeneration.length;
        const sliderMax = total;
        if(!Number.isFinite(raw) || raw >= sliderMax){
          train.historySelection = null;
        } else {
          const idx = Math.min(total - 1, Math.max(0, Math.round(raw)));
          train.historySelection = idx;
        }
      }
      syncHistoryControls();
      updateTrainStatus();
      updateScorePlot();
    };
    if(historySlider){
      historySlider.addEventListener('input', handleHistorySliderInput);
      historySlider.addEventListener('change', handleHistorySliderInput);
    }

    // Hook up training buttons
    const startTrainingBtn = document.getElementById('start-training');
    if(startTrainingBtn){ startTrainingBtn.addEventListener('click', () => { if(train.enabled) stopTraining(); else startTraining(); }); }
    const resetModelBtn = document.getElementById('reset-model');
    if(resetModelBtn){ resetModelBtn.addEventListener('click', resetTraining); }
    const renderToggleInput = document.getElementById('render-toggle');
    const renderToggleLabel = document.getElementById('render-toggle-label');
    function syncRenderToggle(){
      if(!renderToggleInput) return;
      renderToggleInput.checked = !!train.visualizeBoard;
      if(renderToggleLabel){
        renderToggleLabel.dataset.active = renderToggleInput.checked ? 'true' : 'false';
      }
    }
    if(renderToggleInput){
      renderToggleInput.checked = !!train.visualizeBoard;
      renderToggleInput.addEventListener('change', () => {
        train.visualizeBoard = renderToggleInput.checked;
        const populationModel = usesPopulationModel(train.modelType);
        train.plotBestOnly = populationModel && !train.visualizeBoard;
        train.scorePlotUpdateFreq = populationModel
          ? SCORE_PLOT_DEFAULT_UPDATE_FREQ
          : SCORE_PLOT_ALPHATETRIS_UPDATE_FREQ;
        syncRenderToggle();
        if(train.visualizeBoard){
          try { draw(state.grid, state.active); drawNext(state.next); } catch(_) {}
          updateScore(true); updateLevel(true);
        }
        const bestCount = Array.isArray(train.bestByGeneration) ? train.bestByGeneration.length : 0;
        const genCount = Number.isFinite(train.gen) ? train.gen : 0;
        const popBaseline = Math.max(1, Number.isFinite(train.popSize) ? train.popSize : 0);
        const gameBaseline = Math.max(10, Number.isFinite(train.totalGamesPlayed) ? train.totalGamesPlayed : 0);
        const baselineSource = train.plotBestOnly
          ? Math.max(1, bestCount, genCount, populationModel ? 1 : 10)
          : (populationModel ? popBaseline : gameBaseline);
        const baselineAxis = Math.max(10, Math.ceil(baselineSource * 1.2));
        const cap = Math.max(1, train.maxPlotPoints || baselineAxis);
        train.scorePlotAxisMax = Math.min(cap, baselineAxis);
        train.scorePlotPending = 0;
        updateTrainStatus();
        updateScorePlot();
      });
    }
    syncRenderToggle();
    initMlpConfigUi();
    const modelSel = document.getElementById('model-select');
    function setModelType(mt){
      if(mt !== 'linear' && !isMlpModelType(mt) && !isAlphaModelType(mt)) return;
      const wasRunning = train.enabled;
      if(wasRunning) stopTraining();
      const prevModelType = train.modelType;
      train.modelType = mt;
      currentModelType = mt;
      train.mlpHiddenLayers = mlpHiddenLayers.slice();
      if(isAlphaModelType(mt)){
        train.dtype = 'f32';
        dtypePreference = train.dtype;
        train.mean = null;
        train.std = null;
        train.meanView = null;
        train.stdView = null;
        train.candWeights = [];
        train.candWeightViews = [];
        train.candScores = [];
        train.candIndex = -1;
        train.currentWeightsOverride = null;
        train.bestEverWeights = null;
        train.bestEverFitness = -Infinity;
        const prevAlpha = train.alpha || null;
        const prevConfig = prevAlpha && prevAlpha.config ? prevAlpha.config : null;
        train.alpha = createAlphaState(prevConfig);
        if(prevAlpha && prevAlpha.latestModel && prevModelType === 'alphatetris'){
          train.alpha.latestModel = prevAlpha.latestModel;
          train.alpha.modelPromise = prevAlpha.modelPromise || Promise.resolve(prevAlpha.latestModel);
          train.alpha.lastRootValue = prevAlpha.lastRootValue;
          train.alpha.lastPolicyLogits = prevAlpha.lastPolicyLogits;
        }
      } else {
        // Prefer f16 for MLP if available, else f32
        train.dtype = (isMlpModelType(mt) && HAS_F16) ? 'f16' : 'f32';
        dtypePreference = train.dtype;
        // Re-init mean/std to the appropriate initial values for this model
        train.mean = initialMean(mt);
        train.meanView = createDisplayView(train.mean, train.meanView);
        train.std  = initialStd(mt);
        train.stdView = createDisplayView(train.std, train.stdView);
        if(train.alpha && prevModelType === 'alphatetris'){
          disposeAlphaModel(train.alpha);
        }
        train.alpha = null;
      }
      updateTrainStatus();
      // Reset training state
      resetTraining();
      syncMlpConfigVisibility();
      if(wasRunning) startTraining();
    }
    if(modelSel){ modelSel.addEventListener('change', (e) => setModelType(modelSel.value)); }
    if(mctsSimulationInput){
      mctsSimulationInput.addEventListener('change', () => {
        const next = sanitizeSimulationCount(mctsSimulationInput.value, train.ai.search.simulations);
        train.ai.search.simulations = next;
        mctsSimulationInput.value = String(next);
      });
    }
    if(mctsCpuctInput){
      mctsCpuctInput.addEventListener('change', () => {
        const next = sanitizeExplorationConstant(mctsCpuctInput.value, train.ai.search.cPuct);
        train.ai.search.cPuct = next;
        mctsCpuctInput.value = String(next);
      });
    }
    if(mctsTemperatureInput){
      mctsTemperatureInput.addEventListener('change', () => {
        const next = sanitizeTemperature(mctsTemperatureInput.value, train.ai.search.temperature);
        train.ai.search.temperature = next;
        mctsTemperatureInput.value = String(next);
      });
    }
    updateTrainStatus();


  const hooks = {
    aiStep,
    isEnabled: () => train.enabled,
    shouldSkipRendering: () => train.enabled && train.visualizeBoard === false,
  };
  game.registerTrainingInterface(hooks);
  game.setGameOverHandler(onGameOver);
  return { train: window.__train, hooks };
}
