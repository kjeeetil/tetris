# Tetris

This repository provides a minimal Tetris implementation that runs in the browser using [PyScript](https://pyscript.net) and [`pygame-ce`](https://github.com/pygame-community/pygame-ce).

## Requirements

- A modern web browser with JavaScript enabled.
- Internet access to load PyScript and the `pygame-ce` package. When running locally with Python, install dependencies with `pip install -r requirements.txt`.

## Running

1. Open [`index.html`](index.html) in your web browser.
2. PyScript downloads the project and `pygame-ce`, then executes the Python block at the bottom of the page.
3. The block imports `run_pygame` and calls `run_pygame.main()` which initializes the display and enters the main PyGame loop within the page's canvas element.

## Limitations

This front end is intentionally lightweight and only showcases core gameplay mechanics:

- No sound, scoring effects, or advanced Tetris features like hold queue or piece previews.
- Browser performance and keyboard focus may vary.

Additional features may be added in the future.

