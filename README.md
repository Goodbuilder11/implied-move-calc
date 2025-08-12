## Implied Move Calculator

Simple program that estimates a stock's implied move for the next expiration by pricing the at‑the‑money straddle using Alpaca options data. It prints a compact, readable summary.

### Demo
![Demo](./demo.png)

### Requirements

- Alpaca API keys with options market data access

### Setup
1) Create a `.env` file in the project root (or set these in your shell):
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

2) Install dependencies and run.

Option A — uv (recommended):
```
uv sync
uv run implied-move
```

Option B — pip/venv:
```
python -m venv .venv
source .venv/bin/activate
pip install .
implied-move
```

You can also run without installing the script entry point:
```
uv run python implied_move.py
# or
python implied_move.py
```

### Usage
- The program will prompt: `Enter the stock symbol (or 'q' to quit):`
- Expiration defaults to today (next available expiry is chosen if needed).
- If the exact ATM strike is missing, it selects the nearest strike within a ±10% window.

### Notes
- Requires access to Alpaca options data. Some tickers or market states may not have quotes.
- Script entry point installed by this project is `implied-move` (see `pyproject.toml`).

