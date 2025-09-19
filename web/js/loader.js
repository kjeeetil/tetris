const TENSORFLOW_SRC = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.18.0/dist/tf.min.js';
const TENSORFLOW_VIS_SRC = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.5.1/dist/tfjs-vis.umd.min.js';

let tensorflowLoadPromise = null;
let tensorflowVisLoadPromise = null;

export async function ensureTensorFlowLoaded() {
  if (typeof window === 'undefined') {
    throw new Error('TensorFlow.js can only be loaded in a browser environment.');
  }

  if (window.tf) {
    return window.tf;
  }

  if (tensorflowLoadPromise) {
    return tensorflowLoadPromise;
  }

  tensorflowLoadPromise = new Promise((resolve, reject) => {
    let script = document.querySelector(`script[src="${TENSORFLOW_SRC}"]`);

    const handleLoad = (event) => {
      const target = event?.currentTarget ?? script;
      if (target) {
        target.dataset.tensorflowLoaderLoaded = 'true';
      }

      if (window.tf) {
        resolve(window.tf);
      } else {
        const error = new Error('TensorFlow.js loaded but window.tf is unavailable.');
        console.error(error);
        tensorflowLoadPromise = null;
        reject(error);
      }
    };

    const handleError = (event) => {
      console.error('Failed to load TensorFlow.js from CDN.', event);
      tensorflowLoadPromise = null;
      reject(new Error('Failed to load TensorFlow.js from CDN.'));
    };

    if (!script) {
      script = document.createElement('script');
      script.src = TENSORFLOW_SRC;
      script.async = true;
      script.dataset.tensorflowLoader = 'true';
      script.addEventListener('load', handleLoad, { once: true });
      script.addEventListener('error', handleError, { once: true });
      document.head.appendChild(script);
    } else {
      script.dataset.tensorflowLoader = script.dataset.tensorflowLoader ?? 'true';
      script.addEventListener('load', handleLoad, { once: true });
      script.addEventListener('error', handleError, { once: true });

      if (script.dataset.tensorflowLoaderLoaded === 'true' || script.readyState === 'complete') {
        handleLoad();
      }
    }
  });

  return tensorflowLoadPromise;
}

export async function ensureTensorFlowVisLoaded() {
  if (typeof window === 'undefined') {
    throw new Error('TensorFlow.js Vis can only be loaded in a browser environment.');
  }

  if (window.tfvis) {
    return window.tfvis;
  }

  if (tensorflowVisLoadPromise) {
    return tensorflowVisLoadPromise;
  }

  tensorflowVisLoadPromise = new Promise((resolve, reject) => {
    let script = document.querySelector(`script[src="${TENSORFLOW_VIS_SRC}"]`);

    const handleLoad = (event) => {
      const target = event?.currentTarget ?? script;
      if (target) {
        target.dataset.tensorflowVisLoaderLoaded = 'true';
      }

      if (window.tfvis) {
        resolve(window.tfvis);
      } else {
        const error = new Error('TensorFlow.js Vis loaded but window.tfvis is unavailable.');
        console.error(error);
        tensorflowVisLoadPromise = null;
        reject(error);
      }
    };

    const handleError = (event) => {
      console.error('Failed to load TensorFlow.js Vis from CDN.', event);
      tensorflowVisLoadPromise = null;
      reject(new Error('Failed to load TensorFlow.js Vis from CDN.'));
    };

    if (!script) {
      script = document.createElement('script');
      script.src = TENSORFLOW_VIS_SRC;
      script.async = true;
      script.dataset.tensorflowVisLoader = 'true';
      script.addEventListener('load', handleLoad, { once: true });
      script.addEventListener('error', handleError, { once: true });
      document.head.appendChild(script);
    } else {
      script.dataset.tensorflowVisLoader = script.dataset.tensorflowVisLoader ?? 'true';
      script.addEventListener('load', handleLoad, { once: true });
      script.addEventListener('error', handleError, { once: true });

      if (script.dataset.tensorflowVisLoaderLoaded === 'true' || script.readyState === 'complete') {
        handleLoad();
      }
    }
  });

  return tensorflowVisLoadPromise;
}
