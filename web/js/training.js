import { HEIGHT, MAX_AI_STEPS_PER_FRAME, WIDTH } from './constants.js';
import { Piece, SHAPES, UNIQUE_ROTATIONS, canMove, clearRows, emptyGrid, gravityForLevel, lock } from './engine.js';

let randnSpare = null;

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
  const historySlider = document.getElementById('model-history-slider');
  const historyLabel = document.getElementById('model-history-label');
  const historyMeta = document.getElementById('model-history-meta');
  const mlpConfigEl = document.getElementById('mlp-config');
  const mlpHiddenCountSel = document.getElementById('mlp-hidden-count');
  const mlpLayerControlsEl = document.getElementById('mlp-layer-controls');

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
    function isMlpModelType(type) {
      return type === 'mlp' || type === 'mlp_raw';
    }
    function resolveMlpType(type) {
      return type === 'mlp_raw' ? 'mlp_raw' : 'mlp';
    }
    function inputDimForModel(type) {
      return type === 'mlp_raw' ? RAW_FEAT_DIM : FEAT_DIM;
    }
    function featureNamesForModel(type) {
      return type === 'mlp_raw' ? RAW_FEATURE_NAMES : FEATURE_NAMES;
    }
    function modelDisplayName(type) {
      if (type === 'mlp_raw') {
        return 'MLP (board occupancy)';
      }
      if (type === 'mlp') {
        return 'MLP (engineered features)';
      }
      return 'Linear';
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

    function paramDim(){ return isMlpModelType(currentModelType) ? mlpParamDim(mlpHiddenLayers, currentModelType) : FEAT_DIM; }
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
      if(isMlpModelType(model)){
        const dim = mlpParamDim(layers, model);
        const base = new Array(dim).fill(0.0);
        return makeStatsArray(base);
      }
      return makeStatsArray(INITIAL_MEAN_LINEAR_BASE);
    }
    function initialStd(model, layers = mlpHiddenLayers){
      if(isMlpModelType(model)){
        const dim = mlpParamDim(layers, model);
        const base = new Array(dim).fill(0.2);
        return makeStatsArray(base);
      }
      return makeStatsArray(INITIAL_STD_LINEAR_BASE);
    }

    function describeModelArchitecture(){
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

    function describeSnapshotArchitecture(entry){
      if(!entry){
        return describeModelArchitecture();
      }
      const genLabel = Number.isFinite(entry.gen) ? `Gen ${entry.gen}` : 'Saved model';
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
      return `${genLabel} — Architecture unavailable`;
    }

    function formatScore(value){
      if(!Number.isFinite(value)){
        return 'n/a';
      }
      return Math.round(value).toLocaleString();
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
          historyLabel.textContent = `Gen ${entry.gen}`;
        }
      }

      if(historyMeta){
        if(!hasHistory){
          historyMeta.textContent = 'Best-of-generation snapshots will appear as training progresses.';
        } else if(activeIndex === null){
          const latest = train.bestByGeneration[total - 1];
          const info = [];
          if(Number.isFinite(latest.gen)) info.push(`Latest stored: Gen ${latest.gen}`);
          if(Number.isFinite(latest.fitness)) info.push(`Score ${formatScore(latest.fitness)}`);
          if(latest.modelType) info.push(modelDisplayName(latest.modelType));
          historyMeta.textContent = info.length ? info.join(' • ') : 'Snapshot details unavailable.';
        } else {
          const entry = train.bestByGeneration[activeIndex];
          const info = [];
          if(entry.modelType) info.push(modelDisplayName(entry.modelType));
          if(Number.isFinite(entry.fitness)) info.push(`Score ${formatScore(entry.fitness)}`);
          historyMeta.textContent = info.length ? info.join(' • ') : 'Snapshot details unavailable.';
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
        fitness: snapshot.fitness,
        modelType: snapshot.modelType,
        dtype: snapshot.dtype || train.dtype || DEFAULT_DTYPE,
        layerSizes: Array.isArray(snapshot.layerSizes) ? snapshot.layerSizes.slice() : null,
        weights: snapshot.weights,
        scoreIndex: Number.isFinite(snapshot.scoreIndex) ? snapshot.scoreIndex : null,
        recordedAt: snapshot.recordedAt || Date.now(),
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

    function createWeightSnapshot(){
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

    function downloadCurrentWeights(){
      try {
        const snapshot = createWeightSnapshot();
        if(!snapshot){
          log('Weights unavailable for download yet. Start a game or training session first.');
          return;
        }
        const text = JSON.stringify(snapshot, null, 2);
        const blob = new Blob([text], { type: 'text/plain' });
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const safeModel = (snapshot.modelType || 'model').toLowerCase();
        const fileName = `tetris-${safeModel}-weights-${timestamp}.txt`;
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
        log(`Saved ${modelDisplayName(snapshot.modelType)} weights (${snapshot.weights.length} params) to ${fileName}.`);
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
      if(data.modelType !== 'linear' && !isMlpModelType(data.modelType)){
        throw new Error('Snapshot missing model type');
      }
      if(!Array.isArray(data.weights) || !data.weights.length){
        throw new Error('Snapshot missing weights');
      }
      const expectedFeatures = inputDimForModel(data.modelType || 'linear');
      if(data.featureCount && data.featureCount !== expectedFeatures){
        throw new Error(`Snapshot expects ${data.featureCount} features but this build uses ${expectedFeatures}`);
      }
      data.version = 1;
      return data;
    }

    function applyWeightSnapshot(snapshot, context){
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

      const wasRunning = train && train.enabled;
      if(wasRunning){
        stopTraining();
      }

      if(modelSel){
        modelSel.value = modelType;
      }

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
      currentWeightsOverride: null,
      ai: { plan: null, acc: 0, lastSig: '', staleMs: 0 },
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
      scorePlotUpdateFreq: 5,
      scorePlotPending: 0,
      scorePlotAxisMax: 0,
      meanView: null,
      stdView: null,
    };
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
    const simulateResultScratch = { lines: 0, grid: gridScratch, dropRow: 0, clearedRows: clearedRowsScratch, clearedRowCount: 0 };
    window.__train = train;
    // After `train` exists, honor train.dtype for future allocations
    dtypePreference = train.dtype || DEFAULT_DTYPE;
    train.meanView = createDisplayView(train.mean, train.meanView);
    train.stdView = createDisplayView(train.std, train.stdView);
    train.candWeightViews = [];

    train.scorePlotAxisMax = Math.max(10, Math.ceil(train.popSize * 1.2));
    train.scorePlotPending = 0;
    syncHistoryControls();

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
        if(train.enabled){
          trainStatus.textContent = `Gen ${train.gen+1}, Candidate ${train.candIndex+1}/${train.popSize} — Model: ${statusLabel}`;
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
      } else if(train.enabled && train.candIndex >= 0 && train.candIndex < train.candWeights.length){
        currentWeights = train.candWeights[train.candIndex];
        if(isMlpModelType(train.modelType)){
          overrideLayers = currentMlpLayerSizes(train.modelType);
        }
      } else if(train.mean){
        currentWeights = train.mean;
        if(isMlpModelType(train.modelType)){
          overrideLayers = currentMlpLayerSizes(train.modelType);
        }
      }

      if(!snapshot && !currentWeights && train.bestEverWeights){
        currentWeights = train.bestEverWeights;
        overrideLayers = isMlpModelType(train.modelType) ? currentMlpLayerSizes(train.modelType) : [FEAT_DIM, 1];
      }

      const fallbackModelType = train ? train.modelType : currentModelType;
      const displayModelType = snapshot && snapshot.modelType ? snapshot.modelType : fallbackModelType;

      if(architectureEl){
        if(snapshot){
          architectureEl.textContent = describeSnapshotArchitecture(snapshot);
        } else {
          architectureEl.textContent = describeModelArchitecture();
        }
      }
      let displayWeights = currentWeights;
      if(snapshot && currentWeights){
        displayWeights = getDisplayWeightsForUi(currentWeights, { dtype: snapshot.dtype || dtypePreference });
      } else if(currentWeights){
        displayWeights = getDisplayWeightsForUi(currentWeights);
      }
      const vizFeatureNames = featureNamesForModel(displayModelType);
      const vizInputDim = inputDimForModel(displayModelType);
      try {
        renderNetworkD3(displayWeights, overrideLayers, { featureNames: vizFeatureNames, inputDim: vizInputDim });
      } catch (_) {
        /* ignore render failures */
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
      if(!continuing){
        train.performanceSummary = [];
        train.gen = 0;
        train.gameScores = [];
        train.gameModelTypes = [];
        train.gameScoresOffset = 0;
        train.totalGamesPlayed = 0;
        train.scorePlotAxisMax = Math.max(10, Math.ceil(train.popSize * 1.2));
      }
      train.enabled = true;
      resetAiPlanState();
      train.ai.acc = 0;
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
      train.currentWeightsOverride = null;
      train.scorePlotPending = 0;
      updateTrainStatus();
      const btn = document.getElementById('start-training');
      if(btn){
        btn.textContent = 'Stop Training';
        btn.classList.remove('icon-btn--violet');
        btn.classList.add('icon-btn--emerald');
        btn.setAttribute('title', 'Stop training');
        btn.setAttribute('aria-label', 'Stop training');
      }
      if(continuing){
        const totalCandidates = Number.isFinite(train.popSize) && train.popSize > 0
          ? train.popSize
          : ((Array.isArray(train.candWeights) && train.candWeights.length > 0) ? train.candWeights.length : 1);
        const candidateNumber = Math.max(
          1,
          Math.min(totalCandidates, (Number.isFinite(train.candIndex) ? Math.floor(train.candIndex) : 0) + 1)
        );
        log(`Training resumed — Gen ${train.gen + 1}, Candidate ${candidateNumber}/${totalCandidates}`);
      } else {
        log('Training started');
      }
    }
    function stopTraining(){
      const wasRunning = train.enabled;
      train.enabled = false;
      resetAiPlanState();
      const btn = document.getElementById('start-training');
      if(btn){
        btn.textContent = 'Start Training';
        btn.classList.remove('icon-btn--emerald');
        btn.classList.add('icon-btn--violet');
        btn.setAttribute('title', 'Start training');
        btn.setAttribute('aria-label', 'Start training');
      }
      if(train.scorePlotPending && train.gameScores.length){
        updateScorePlot();
      }
      train.scorePlotPending = 0;
      log('Training stopped');
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
      // Reset mean/std based on selected model
      train.mean = initialMean(train.modelType);
      train.std = initialStd(train.modelType);
      train.meanView = createDisplayView(train.mean, train.meanView);
      train.stdView = createDisplayView(train.std, train.stdView);
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
      train.scorePlotAxisMax = Math.max(10, Math.ceil(train.popSize * 1.2));
      updateScorePlot();
      syncHistoryControls();
      updateTrainStatus();
      log('Training parameters reset');
    }
    window.startTraining = startTraining; window.stopTraining = stopTraining; window.resetTraining = resetTraining;

    function resetAiPlanState(){
      if(!train || !train.ai){
        return;
      }
      train.ai.plan = null;
      train.ai.staleMs = 0;
      train.ai.lastSig = '';
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
        if(train.candIndex >= 0) train.candScores[train.candIndex] = fitness;
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
        const updateStride = Math.max(1, train.scorePlotUpdateFreq || 5);
        train.scorePlotPending = (train.scorePlotPending || 0) + 1;
        if(train.scorePlotPending >= updateStride){
          updateScorePlot();
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
    function lockSim(grid, piece){ for(const [r,c] of piece.blocks()) grid[r][c]=1; }
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
    const BOARD_AREA = WIDTH * HEIGHT;
    const BUMP_NORMALIZER = ((WIDTH - 1) * HEIGHT) || 1;
    const CONTACT_NORMALIZER = (BOARD_AREA * 2) || 1;
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

    function dot(weights, feats){ let s=0; for(let d=0; d<FEAT_DIM; d++) s+=weights[d]*feats[d]; return s; }
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
    function choosePlacement(weights, grid, curShape){
      return trainingProfiler.section('train.plan.single', () => {
        const acts = enumeratePlacements(grid, curShape);
        if(acts.length === 0) return null;
        const baselineMetrics = computeGridMetrics(grid);
        const baselineHoles = baselineMetrics ? baselineMetrics.holes : 0;
        for(let c = 0; c < WIDTH; c += 1){
          baselineColumnMaskScratch[c] = columnMaskScratch[c] || 0;
          baselineColumnHeightScratch[c] = columnHeightScratch[c] || 0;
        }
        let best = null;
        let bestScore = -Infinity;
        // Evaluate each valid placement exactly once.
        for(const a of acts){
          const sim = simulateAfterPlacement(grid, curShape, a.rot, a.col);
          if(!sim) continue;
          const lines = sim.lines;
          const dropRow = sim.dropRow;
          const baseFeats = isMlpModelType(train.modelType) && train.modelType === 'mlp_raw'
            ? rawFeaturesFromGrid(sim.grid)
            : featuresFromGrid(sim.grid, lines, {
                holeBaseline: baselineHoles,
                baselineColumnMasks: baselineColumnMaskScratch,
                clearedRows: sim.clearedRows,
              });
          const score = scoreFeats(weights, baseFeats);
          a.dropRow = dropRow;
          a.lines = lines;
          if(score > bestScore){
            bestScore = score;
            best = a;
          }
        }
        return best;
      });
    }
    function planForCurrentPiece(){
      return trainingProfiler.section('train.plan', () => {
        if(!state.active) return null;
        const w = train.currentWeightsOverride || train.candWeights[train.candIndex] || train.mean;
        // Single-ply evaluation selects the best placement for the current piece.
        const placement = choosePlacement(w, state.grid, state.active.shape);
        if(!placement){
          return null;
        }
        const len = SHAPES[state.active.shape].length;
        const cur = state.active.rot % len;
        const needRot = (placement.rot - cur + len) % len;
        const targetRow = Number.isFinite(placement.dropRow) ? placement.dropRow : null;
        return { targetRot: placement.rot, targetCol: placement.col, targetRow, rotLeft: needRot, stage: 'rotate' };
      });
    }

    // Scatter plot of raw score per game (all candidates)
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
    const scores = (trainState && trainState.gameScores) ? trainState.gameScores : [];
    const types  = (trainState && trainState.gameModelTypes) ? trainState.gameModelTypes : [];
    const xw = Math.max(0, W - padL - padR);
    const yh = Math.max(0, H - padT - padB);

    const maxScore = scores.length ? Math.max(...scores) : 0;
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

    const count = scores.length;
    if(trainState && typeof trainState.scorePlotPending !== 'number'){
      trainState.scorePlotPending = 0;
    }
    if(!count){
      if(trainState){
        trainState.scorePlotPending = 0;
        if(!trainState.scorePlotAxisMax || trainState.scorePlotAxisMax < 1){
          const baseline = Math.max(10, Math.ceil((trainState.popSize || 10) * 1.2));
          const maxCap = Math.max(1, trainState.maxPlotPoints || baseline);
          trainState.scorePlotAxisMax = Math.min(maxCap, baseline);
        }
      }
      return;
    }

    let axisMax = count;
    if(trainState){
      const maxCap = Math.max(count, trainState.maxPlotPoints || count);
      let currentAxis = Number.isFinite(trainState.scorePlotAxisMax) ? trainState.scorePlotAxisMax : 0;
      if(currentAxis < 1){
        const baseline = Math.max(10, Math.ceil((trainState.popSize || count || 5) * 1.2));
        currentAxis = Math.min(maxCap, baseline);
      }
      if(count > currentAxis){
        let next = Math.ceil(currentAxis * 1.2);
        if(!Number.isFinite(next) || next <= currentAxis){
          next = currentAxis + 1;
        }
        currentAxis = Math.min(maxCap, Math.max(next, count));
      }
      trainState.scorePlotAxisMax = currentAxis;
      trainState.scorePlotPending = 0;
      axisMax = Math.max(count, currentAxis);
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
    tickSet.add(count);
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
    for(let i=0; i<count; i++){
      const gameNumber = i + 1;
      const ratio = axisMax <= 1 ? 1 : (gameNumber - 1) / denom;
      const x = padL + ratio * xw;
      const y = H - padB - (scores[i] / safeMaxY) * yh;
      const color = COLORS[types[i] || 'linear'] || COLORS.linear;
      pointPositions.push({ x, y });
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(12, 17, 32, 0.85)';
      ctx.stroke();
    }

    // Draw a centered rolling average (last 50 and next 50 games).
    const windowRadius = 50;
    const requiredWindow = windowRadius * 2 + 1;
    if(count >= requiredWindow){
      const prefix = new Array(count + 1);
      prefix[0] = 0;
      for(let i = 0; i < count; i++){
        prefix[i + 1] = prefix[i] + scores[i];
      }
      const startIdx = windowRadius;
      const endIdx = count - windowRadius - 1;
      if(startIdx <= endIdx){
        ctx.save();
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.strokeStyle = '#ffffff';
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        let drewPoint = false;
        for(let i = startIdx; i <= endIdx; i++){
          const windowStart = i - windowRadius;
          const windowEnd = i + windowRadius;
          const windowCount = windowEnd - windowStart + 1;
          if(windowCount <= 0) continue;
          const sum = prefix[windowEnd + 1] - prefix[windowStart];
          const avg = sum / windowCount;
          const point = pointPositions[i];
          if(!point) continue;
          const y = H - padB - (avg / safeMaxY) * yh;
          if(!drewPoint){
            ctx.moveTo(point.x, y);
            drewPoint = true;
          } else {
            ctx.lineTo(point.x, y);
          }
        }
        if(drewPoint){
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    if(trainState && Array.isArray(trainState.bestByGeneration) && trainState.bestByGeneration.length){
      const offset = Number.isFinite(trainState.gameScoresOffset) ? trainState.gameScoresOffset : 0;
      let selection = trainState.historySelection;
      if(selection !== null && selection !== undefined){
        selection = Math.max(0, Math.min(trainState.bestByGeneration.length - 1, Math.round(selection)));
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
              log(topOutLog);
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
          log('AI: watchdog forced drop');
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
              log('AI: top-out after forced drop');
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
              log('AI: rotation blocked, abandoning plan');
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
            log('AI: top-out after drop');
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
      if(!canMove(state.grid, state.active, 0, 0)) { log('AI: spawn blocked -> game over'); onGameOver(); return; }

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
      downloadBtn.addEventListener('click', downloadCurrentWeights);
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
        reader.onload = (event) => {
          try {
            const text = typeof event.target.result === 'string' ? event.target.result : '';
            const snapshot = parseWeightSnapshot(text);
            applyWeightSnapshot(snapshot, { fileName: file.name });
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
        syncRenderToggle();
        if(train.visualizeBoard){
          try { draw(state.grid, state.active); drawNext(state.next); } catch(_) {}
          updateScore(true); updateLevel(true);
        }
      });
    }
    syncRenderToggle();
    initMlpConfigUi();
    const modelSel = document.getElementById('model-select');
    function setModelType(mt){
      if(mt !== 'linear' && mt !== 'mlp' && mt !== 'mlp_raw') return;
      const wasRunning = train.enabled;
      if(wasRunning) stopTraining();
      train.modelType = mt;
      currentModelType = mt;
      train.mlpHiddenLayers = mlpHiddenLayers.slice();
      // Prefer f16 for MLP if available, else f32
      train.dtype = (isMlpModelType(mt) && HAS_F16) ? 'f16' : 'f32';
      dtypePreference = train.dtype;
      // Re-init mean/std to the appropriate initial values for this model
      train.mean = initialMean(mt);
      train.meanView = createDisplayView(train.mean, train.meanView);
      train.std  = initialStd(mt);
      train.stdView = createDisplayView(train.std, train.stdView);
      updateTrainStatus();
      // Reset training state
      resetTraining();
      syncMlpConfigVisibility();
      if(wasRunning) startTraining();
    }
    if(modelSel){ modelSel.addEventListener('change', (e) => setModelType(modelSel.value)); }
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
