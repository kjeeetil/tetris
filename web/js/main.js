import { ensureTensorFlowLoaded } from './loader.js';
import { createRenderer } from './rendering.js';
import { createGame } from './game-loop.js';
import { initTraining } from './training.js';

function loadTensorFlow(renderer) {
  return ensureTensorFlowLoaded()
    .then(() => {
      if (renderer && typeof renderer.log === 'function') {
        renderer.log('TensorFlow.js loaded. Training features ready.');
      }
      return { ok: true };
    })
    .catch((error) => {
      console.warn('TensorFlow.js failed to load; training features are disabled.', error);
      if (renderer && typeof renderer.log === 'function') {
        renderer.log('TensorFlow.js failed to load. Training features are disabled.');
      }
      return { ok: false, error };
    });
}

function bootstrap() {
  try {
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
    const tensorflowReady = loadTensorFlow(renderer);
    initTraining(game, renderer, { tensorflowReady });
  } catch (error) {
    console.error('Failed to bootstrap application.', error);
  }
}

bootstrap();
