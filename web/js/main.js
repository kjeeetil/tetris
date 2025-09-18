import { ensureTensorFlowLoaded } from './loader.js';
import { createRenderer } from './rendering.js';
import { createGame } from './game-loop.js';
import { initTraining } from './training.js';

async function bootstrap() {
  try {
    await ensureTensorFlowLoaded();

    const canvas = document.getElementById('canvas');
    const preview = document.getElementById('preview');
    const diagnostics = document.getElementById('diagnostics');
    const scoreEl = document.getElementById('score');
    const levelEl = document.getElementById('level');
    const toggleBtn = document.getElementById('toggle');
    const resetBtn = document.getElementById('reset');
    const speedSlider = document.getElementById('speed');
    const speedDisplay = document.getElementById('speed-display');

    const renderer = createRenderer({ canvas, preview, diagnostics, scoreEl, levelEl });
    const game = createGame({ canvas, toggleBtn, resetBtn, speedSlider, speedDisplay }, renderer);
    initTraining(game, renderer);
  } catch (error) {
    console.error('Failed to bootstrap application.', error);
  }
}

bootstrap();
