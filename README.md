# Tetris

## Overview

This project packages a playable Tetris clone together with a lightweight
reinforcement-learning lab. The web front end renders the game with
vanilla JavaScript and HTML canvas while exposing controls that let you train
heuristic agents directly in the browser. The `src/tetris` Python package mirrors
the same rules and placement logic so you can export experiments to Gym-compatible
workflows or offline research scripts.

## Site build-up

### Front-end stack

- Single-page application served from `web/index.html` with no build tooling –
  all styling and layout use Tailwind via CDN, plus Google Fonts for typography.
- Canvas-based renderer backed by a compact `Piece` class and board helpers for
  locking, row clearing, previews, and scoring updates.
- Optional visualisations supplied by D3.js (for network diagrams) and the
  Canvas API (for score charts and diagnostics).
- TensorFlow.js is loaded for experimentation with neural policies; the default
  inference path keeps evaluation in plain JavaScript for performance.

### Layout and interaction flow

- The main column hosts the playfield canvas, next-piece preview, level and score
  readouts, and tactile play/reset buttons.
- A second control strip enables AI-specific actions: start/stop training,
  reset evolution, pick between linear or MLP models, toggle headless training,
  and throttle evaluation speed.
- Analytics panels underneath render the scatter plot of per-game scores, an
  interactive network graph of the active policy, and a rolling diagnostics log
  that captures watchdog resets, training events, and game-over messages.

## Gameplay and engine mechanics

- Standard 10×20 matrix with 7-bag random piece generation ensures fair
  distribution of tetrominoes and mirrors competitive rule sets.
- Gravity timings follow the NES level table up to level nine; levels increase
  every 20 pieces and adjust the millisecond delay between drops.
- Scoring awards 100 points per cleared line with a multiplier for clearing
  multiple rows simultaneously; hard drops via the space bar instantly lock the
  piece and process line clears.
- A watchdog monitors the active piece signature and restarts the round if no
  progress occurs for two real-time seconds, protecting the loop from animation
  stalls or stuck AI actions.
- Valid placements are enumerated with a gravity-aware feasibility check so the
  AI only chooses moves reachable under natural falling motion; the same logic is
  reused in the Python placement environment through cached bitboard searches.

## Training mechanics and theory

- Each candidate move is scored by simulating the resulting board and computing
  handcrafted features: cleared-line count, indicator flags for singles through
  tetrises, normalized hole count, column bumpiness, maximum stack height,
  well depth, edge wells, contact area, row/column transitions, and aggregate
  height.
- Policies can be linear or a tiny one-hidden-layer MLP (ReLU activations). Both
  operate on the 15-dimensional feature vector; the MLP runs with hand-written
  matrix multiplies to avoid tf.js overhead during search.
- A shallow beam-search lookahead (depth 1 with next-piece preview) combines the
  immediate feature score with a discounted estimate (`λ = 0.7`) of the best
  follow-up placement, helping the agent prefer sustainable surfaces.
- Training uses evolution strategies with antithetic sampling, per-dimension
  standard deviations, and heavy-tailed exploration for a fraction of the
  population. Rank-weighted elites update the search distribution, and the best
  candidate from each generation is re-evaluated multiple times to reduce noise.
- Fitness blends the in-game score (scaled by 3×) with shaped rewards that
  encourage multi-line clears and penalise singles, yielding faster convergence
  towards policies that stack for tetrises.

## Python toolkit

- `src/tetris/placement_env.py` implements the same placement-level decision loop
  found in the browser, including gravity-aware reachability, scoring, and
  optional deterministic 7-bag generation for reproducible experiments.
- `src/tetris/bitboard.py` accelerates placement enumeration by caching bitmask
  representations of board states and precomputed rotation metadata, reducing the
  cost of feasibility checks during search.
- `src/tetris/gym_env.py` exposes a Gymnasium-compatible wrapper that converts
  placement observations into flat vectors with optional action masks for RL
  libraries such as Stable-Baselines3.
- Example scripts in `examples/` demonstrate profiling of placement caches and a
  PPO training stub that integrates with the Gym wrapper.

## Running the site

1. Open `web/index.html` directly in any modern browser; no build or bundler is
   required.
2. Click **Start** to begin a manual game. Arrow keys move, **Up** rotates,
   **Space** hard drops.
3. Use the AI control buttons to launch evolutionary training, switch models,
   or toggle headless evaluation.

## Further exploration

- Install the Python dependencies from `requirements.txt` to experiment with the
  Gym environment or integrate the engine into custom trainers.
- The code is intentionally modular: extend the feature vector, plug in tf.js
  optimisers, or swap the fitness shaping logic to explore alternate heuristics.
