# Tetris

## Overview

This project packages a playable Tetris clone together with a lightweight
reinforcement-learning lab. The web front end still renders the game with
vanilla JavaScript and HTML canvas, but the gameplay, rendering, and training
logic now live in dedicated ES modules under `web/js/`. The `index.html` file is
a slim shell that stitches those modules together, keeping the experience
bundler-free while making the codebase easier to navigate.

## Project layout

### Web front end

- `web/index.html` – Static markup, analytics panels, and script/style tags that
  load the rest of the client without a build step.
- `web/styles.css` – Custom gradients, layout polish, and component styling that
  complement Tailwind utility classes.
- `web/tailwind-config.js` – Runtime configuration passed to the CDN Tailwind
  loader so typography and the colour palette stay consistent across modules.
- `web/js/constants.js` – Shared board dimensions, drop timings, colour tokens,
  and other configuration flags.
- `web/js/engine.js` – `Piece` class plus pure helpers for bag randomisation,
  collision tests, gravity lookups, locking pieces, and clearing rows.
- `web/js/rendering.js` – Canvas rendering pipeline, responsive board scaling,
  HUD updates, and diagnostics logging.
- `web/js/game-loop.js` – Input handling, pause/play state machine, watchdog,
  scoring logic, and the hooks that let the trainer drive the game.
- `web/js/training.js` – Evolution-strategy trainer, weight persistence,
  snapshot history slider, TensorFlow.js integration, and the UI wiring for the
  model controls.

### Infrastructure

- `Dockerfile` – Builds the static site into an Nginx image for easy hosting.
- `cloudbuild.yaml` – Sample Cloud Build recipe that uses the Dockerfile to
  publish the site.

## Front-end stack

- Modular ES modules loaded directly in the browser; no bundler or transpiler is
  required for development.
- Tailwind CSS served via CDN with a minimal runtime config, plus handcrafted
  CSS for the neon glassmorphism aesthetic.
- Canvas-based renderer backed by a compact `Piece` class and board helpers for
  locking, row clearing, previews, and scoring updates.
- Optional visualisations supplied by D3.js (for network diagrams) and the
  Canvas API (for score charts and diagnostics).
- TensorFlow.js is available for experimenting with neural policies; the default
  inference path keeps evaluation in plain JavaScript for performance.

## Layout and interaction flow

- **Main play surface** – The central column holds the game canvas, next-piece
  preview, and live level/score readouts. Manual play uses the keyboard
  (arrow keys to move, `↑` to rotate, `Space` to hard drop) and on-screen play
  and reset buttons.
- **Game pacing controls** – A speed slider scales the gravity delay, while the
  toggle button can pause/resume whether the AI or a human is in control.
- **Training console** – Buttons launch or stop evolutionary training, reset the
  search distribution, switch between linear and MLP policies, toggle headless
  training renders, and manage weight import/export.
- **Model configuration** – When the MLP policy is selected, a dedicated panel
  exposes hidden-layer counts and per-layer width selectors. A history slider
  lets you revisit best-of-generation snapshots, and status text tracks the
  active generation alongside timing stats.
- **Analytics** – Score history plots, a network graph, and a rolling diagnostics
  log capture watchdog events, game overs, and trainer telemetry.

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
  AI only chooses moves reachable under natural falling motion.

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

## Running the site

1. From the repository root, change into the `web/` directory.
2. Serve the directory with any static web server (for example,
   `python3 -m http.server 8000`). Using HTTP avoids module-loading and
   Content-Security-Policy issues that can appear when opening files directly.
3. Visit `http://localhost:8000` in a modern browser, start the game, and use the
   control clusters to play manually or launch the trainer.

## Further exploration

- The code is intentionally modular: extend the feature vector, plug in tf.js
  optimisers, or swap the fitness shaping logic to explore alternate heuristics.
- Because the modules are decoupled, you can experiment with new renderers or
  game inputs by editing `rendering.js` and `game-loop.js` without touching the
  trainer internals.
