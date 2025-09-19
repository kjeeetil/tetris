const EPSILON = 1e-6;

function safeNumber(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

export function createNode(options = {}) {
  const node = {
    prior: Math.max(EPSILON, safeNumber(options.prior, 1)),
    reward: safeNumber(options.reward, 0),
    parent: options.parent || null,
    visitCount: 0,
    valueSum: 0,
    valueEstimate: safeNumber(options.valueEstimate, 0),
    children: new Map(),
    expanded: Boolean(options.expanded),
    stateKey: options.stateKey || null,
    metadata: options.metadata || null,
  };
  return node;
}

export function childQValue(child, discount = 1) {
  if (!child) {
    return 0;
  }
  const baseValue = child.visitCount > 0
    ? child.valueSum / Math.max(1, child.visitCount)
    : safeNumber(child.valueEstimate, 0);
  const reward = safeNumber(child.reward, 0);
  return reward + discount * baseValue;
}

function ucbScore(parent, child, cPuct, discount = 1) {
  if (!child) return -Infinity;
  const totalVisits = parent ? parent.visitCount : 0;
  const q = childQValue(child, discount);
  const u = cPuct * child.prior * Math.sqrt((totalVisits || 0) + 1) / (1 + child.visitCount);
  return q + u;
}

export function selectChild(node, cPuct, discount = 1) {
  if (!node || !node.children || node.children.size === 0) {
    return [null, null];
  }
  let bestScore = -Infinity;
  let bestEntry = null;
  for (const entry of node.children.entries()) {
    const [action, child] = entry;
    const score = ucbScore(node, child, cPuct, discount);
    if (score > bestScore) {
      bestScore = score;
      bestEntry = entry;
    }
  }
  return bestEntry || [null, null];
}

export function backpropagate(path, leafValue, discount = 1) {
  let value = safeNumber(leafValue, 0);
  for (let i = path.length - 1; i >= 0; i -= 1) {
    const entry = path[i];
    if (!entry || !entry.node) {
      continue;
    }
    const node = entry.node;
    node.visitCount += 1;
    node.valueSum += value;
    const reward = safeNumber(entry.reward, 0);
    value = reward + discount * value;
  }
}

export function visitStats(node, discount = 1) {
  const stats = [];
  if (!node || !node.children) {
    return stats;
  }
  for (const [key, child] of node.children.entries()) {
    stats.push({
      key,
      visits: child.visitCount,
      prior: child.prior,
      value: childQValue(child, discount),
      reward: child.reward,
      metadata: child.metadata || null,
    });
  }
  return stats;
}

export function computeVisitPolicy(stats, temperature = 1) {
  if (!Array.isArray(stats) || stats.length === 0) {
    return [];
  }
  const temp = Math.max(0, safeNumber(temperature, 1));
  if (temp === 0) {
    let best = stats[0];
    for (let i = 1; i < stats.length; i += 1) {
      if ((stats[i].visits || 0) > (best.visits || 0)) {
        best = stats[i];
      }
    }
    return stats.map((entry) => ({ ...entry, policy: entry === best ? 1 : 0 }));
  }
  const adjusted = stats.map((entry) => {
    const count = Math.max(0, entry.visits || 0);
    const powered = Math.pow(count, 1 / temp);
    return { entry, powered };
  });
  let sum = 0;
  for (const item of adjusted) {
    sum += item.powered;
  }
  if (sum <= 0) {
    const uniform = 1 / stats.length;
    return stats.map((entry) => ({ ...entry, policy: uniform }));
  }
  return adjusted.map((item) => ({ ...item.entry, policy: item.powered / sum }));
}

export function sampleFromPolicy(statsWithPolicy, rng = Math.random) {
  if (!Array.isArray(statsWithPolicy) || statsWithPolicy.length === 0) {
    return null;
  }
  let total = 0;
  for (const entry of statsWithPolicy) {
    total += entry.policy || 0;
  }
  if (total <= 0) {
    const idx = Math.floor(rng() * statsWithPolicy.length) % statsWithPolicy.length;
    return statsWithPolicy[idx] || null;
  }
  const target = rng() * total;
  let acc = 0;
  for (const entry of statsWithPolicy) {
    acc += entry.policy || 0;
    if (acc >= target) {
      return entry;
    }
  }
  return statsWithPolicy[statsWithPolicy.length - 1] || null;
}

export function softmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) {
    return [];
  }
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = safeNumber(logits[i], -Infinity);
    if (v > maxLogit) {
      maxLogit = v;
    }
  }
  const exps = new Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const v = Math.exp(safeNumber(logits[i], maxLogit) - maxLogit);
    exps[i] = v;
    sum += v;
  }
  if (sum === 0) {
    const uniform = 1 / logits.length;
    return exps.map(() => uniform);
  }
  return exps.map((v) => v / sum);
}

export function normalizePriors(actions) {
  if (!Array.isArray(actions) || actions.length === 0) {
    return actions;
  }
  let sum = 0;
  for (const action of actions) {
    sum += Math.max(0, safeNumber(action.prior, 0));
  }
  if (sum <= 0) {
    const uniform = 1 / actions.length;
    for (const action of actions) {
      action.prior = uniform;
    }
    return actions;
  }
  for (const action of actions) {
    const prior = Math.max(0, safeNumber(action.prior, 0));
    action.prior = prior / sum;
  }
  return actions;
}
