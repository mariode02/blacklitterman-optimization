# Black-Litterman Portfolio Optimization CLI

A command-line tool that combines machine learning forecasts with the Black-Litterman model to produce optimized portfolio allocations. ML models (Random Forest or Ridge Regression) generate return views from technical features, which are blended with a CAPM equilibrium prior via the Black-Litterman framework, then passed to a convex mean-risk optimizer.

## How It Works

1. **Data** — Historical prices and volumes are fetched via `yfinance` (`auto_adjust=True`, `repair=True`)
2. **Features** — 8 technical signals per asset: 1m/3m/6m/12m momentum, 21d volatility, 14d RSI, volume trend, MA crossover
3. **Views** — ML model predicts 21-day forward returns; predictions become Black-Litterman absolute view strings
4. **Prior** — `EquilibriumMu` (CAPM) + `LedoitWolf` covariance as the equilibrium prior
5. **Optimization** — `skfolio` `MeanRisk` maximizes the Sharpe ratio subject to weight constraints

## Installation

```bash
git clone https://github.com/mariode02/blacklitterman-optimization.git
cd blacklitterman-optimization

# Create and activate conda environment
conda create -n bl-optimizer python=3.11 -y
conda activate bl-optimizer

# Install project and dev dependencies
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Input Format

Create a CSV file with your universe of assets:

```csv
ticker,market_cap_weight
AAPL,0.30
MSFT,0.25
GOOGL,0.20
AMZN,0.15
META,0.10
```

`market_cap_weight` is optional — if omitted, equal weights are used for the equilibrium prior.

## Usage

### Run optimization

```bash
bl-optimize run --input tickers.csv --start 2018-01-01 --model rf
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to tickers CSV |
| `--start` | `2018-01-01` | History start date (ISO 8601) |
| `--end` | today | History end date |
| `--model` | `rf` | `rf` = Random Forest, `lr` = Ridge Regression |
| `--objective` | `maximize-ratio` | `maximize-ratio`, `maximize-return`, `minimize-risk` |
| `--output` | — | If set, writes weights to this CSV path |

### Validate data availability

```bash
bl-optimize validate --input tickers.csv
```

### Walk-forward backtest

```bash
bl-optimize backtest --input tickers.csv --start 2018-01-01 --model rf
```

## Configuration

All hyperparameters are configurable via environment variables (prefix `BL_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `BL_RISK_FREE_RATE` | `0.04` | Annual risk-free rate |
| `BL_RISK_AVERSION` | `1.0` | EquilibriumMu lambda |
| `BL_TAU` | `0.05` | Black-Litterman tau scalar |
| `BL_MIN_WEIGHT` | `0.0` | Minimum weight per asset |
| `BL_MAX_WEIGHT` | `0.40` | Maximum weight per asset |
| `BL_CORRELATION_DROP_THRESHOLD` | `0.90` | Drop assets with pairwise correlation above this |
| `BL_MIN_HISTORY_DAYS` | `252` | Minimum trading days required per asset |

## Project Structure

```
src/bl_optimizer/
├── cli/           # Typer commands (run, validate, backtest)
├── config/        # Pydantic settings (BL_* env vars)
├── data/          # yfinance fetcher and validator
├── features/      # Technical feature engineering
├── views/         # ML models and BL view builder
├── optimization/  # skfolio pipeline (DropCorrelated + MeanRisk)
└── reporting/     # Rich tables and metrics panel
```

## Dependencies

- [`skfolio`](https://skfolio.org) — portfolio optimization
- [`yfinance`](https://github.com/ranaroussi/yfinance) — market data
- [`scikit-learn`](https://scikit-learn.org) — ML models
- [`typer`](https://typer.tiangolo.com) + [`rich`](https://rich.readthedocs.io) — CLI interface
- [`pydantic`](https://docs.pydantic.dev) — validation and settings
