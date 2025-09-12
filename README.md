# Tetris

This repository provides a minimal Tetris implementation:

- Web: Pure JavaScript renderer (no dependencies) served from `index.html` or `web/index.html`.
- Desktop: Python + `pygame-ce` engine (`python -m tetris.run_pygame`).

## Requirements

- A modern web browser with JavaScript enabled.
- Internet access to load PyScript and the `pygame-ce` package. When running locally with Python, install dependencies with `pip install -r requirements.txt`.

## Running

1. Open [`index.html`](index.html) or `web/index.html` in your browser (no network CDNs required).
2. Click Start. Use arrow keys to move, Up to rotate, Space to hard‑drop.

## Limitations

This front end is intentionally lightweight and only showcases core gameplay mechanics:

- No sound, scoring effects, or advanced Tetris features like hold queue or piece previews.
- Browser performance and keyboard focus may vary.

Additional features may be added in the future.

