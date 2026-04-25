# Twitter Sentiment vs Cryptocurrency Prices

[![CI](https://github.com/iragoum/crypto-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/iragoum/crypto-sentiment-analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A statistical study of the relationship between **Twitter sentiment** and the price dynamics of **Bitcoin (BTC)**, **Ethereum (ETH)**, and **Binance Coin (BNB)** — using both lexicon-based (VADER) and transformer-based (FinBERT) sentiment analysis, combined with rigorous time-series statistics.

> Bachelor thesis — Peoples' Friendship University of Russia (RUDN), 2026  
> Department of Mathematical Modelling and Artificial Intelligence  
> Direction 02.03.02 "Fundamental Informatics and Information Technologies"

---

## Overview

| | |
|---|---|
| **Dataset** | 703 536 filtered tweets (BTC: 321 627 · ETH: 306 738 · BNB: 75 171) |
| **Period** | 26 Jul – 30 Aug 2022 · 36 days |
| **Sentiment models** | VADER (lexicon) + FinBERT `ProsusAI/finbert` (transformer, 109.5 M params) |
| **Statistical tests** | Pearson/Spearman correlation · Granger causality (raw + differenced) · ADF stationarity · Post-hoc power analysis |
| **Classifiers** | Logistic Regression · Gradient Boosting · DummyClassifier baseline |
| **Validation** | Walk-forward cross-validation · Bootstrap 95 % CI (1 000 iterations) |
| **Tests** | 354 automated tests · Python 3.10 / 3.11 / 3.12 · GitHub Actions |

---

## Repository Structure

```
.
├── code/
│   ├── .github/workflows/ci.yml   # CI matrix: Python 3.10 / 3.11 / 3.12
│   ├── src/
│   │   ├── data_loader.py          # Load & validate filtered tweet CSVs
│   │   ├── preprocessor.py         # Text cleaning, dual-format output (VADER / FinBERT)
│   │   ├── sentiment_analyzer.py   # VADER scoring + FinBERT inference (CPU)
│   │   ├── price_fetcher.py        # Binance Vision → CoinGecko → fallback price chain
│   │   ├── binance_loader.py       # Aggregate 1-min OHLCV → daily prices
│   │   ├── correlation_analyzer.py # Pearson/Spearman, lagged corr, Granger, ADF
│   │   ├── power_analysis.py       # Fisher z post-hoc power, min detectable r
│   │   └── predictor.py            # Binary return classifier, walk-forward CV, bootstrap CI
│   ├── tests/                      # 354 pytest tests
│   ├── data/processed/             # Cached daily prices: BTC / ETH / BNB (committed)
│   ├── results/
│   │   ├── btc/   figures/ + tables/
│   │   ├── eth/   figures/ + tables/
│   │   ├── bnb/   figures/ + tables/
│   │   ├── cross_crypto/           # 3-asset comparison tables + figures
│   │   ├── prediction/             # Classifier metrics per asset + panel
│   │   └── finbert_torchinfo.txt   # FinBERT architecture summary (torchinfo)
│   ├── main.py                     # 12-step end-to-end pipeline
│   ├── convert_data.py             # Filter raw Kaggle CSVs → per-asset datasets
│   ├── requirements.txt
│   └── requirements-test.txt
└── README.md
```

---

## Pipeline

`main.py` runs a **12-step sequential pipeline**:

```
Step  1  Load           Read filtered tweets for selected crypto(s)
Step  2  Preprocess     Clean text; produce original (VADER) + cleaned (FinBERT) variants
Step  3  VADER          Score sentiment with lexicon model → compound ∈ [-1, +1]
Step  4  FinBERT        3-class inference (positive / neutral / negative) on CPU
Step  5  torchinfo      Print FinBERT architecture summary
Step  6  Prices         Load daily OHLCV from Binance Vision cache / CoinGecko / fallback
Step  7  Aggregate      Daily sentiment stats merged with price returns
Step  8  Correlation    Pearson & Spearman (lag 0–7), correlation matrix
Step  9  Granger (raw)  Granger causality on original series (lag 1–4)
Step 10  Granger (diff) ADF test → first difference if needed → Granger on stationary series
Step 11  Power          Post-hoc power for observed r; minimum detectable r at 80 % power
Step 12  Predictor      LR / GBM / Dummy classifier, walk-forward CV, bootstrap 95 % CI
```

---

## Installation

```bash
git clone https://github.com/iragoum/crypto-sentiment-analysis.git
cd crypto-sentiment-analysis/code

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# CPU-only PyTorch (no GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# All dependencies
pip install -r requirements.txt
```

> FinBERT (`ProsusAI/finbert`) downloads automatically on first run via HuggingFace Hub (~450 MB).  
> No GPU needed — the entire pipeline runs on CPU.

### Dataset

Raw tweets are **not included** (download from Kaggle):  
[kaggle.com/datasets/ilariamazzoli/3-million-tweets-cryptocurrencies-btc-eth-bnb](https://www.kaggle.com/datasets/ilariamazzoli/3-million-tweets-cryptocurrencies-btc-eth-bnb)

Unzip to `code/data/raw/`, then filter:

```bash
python convert_data.py --crypto all   # produces tweets_{btc,eth,bnb}.csv
```

Daily price files are already committed under `code/data/processed/`.

---

## Usage

```bash
# Full pipeline — all three assets (VADER + FinBERT)
python main.py --crypto all

# VADER only, fast (~5 min)
python main.py --crypto all --skip-finbert

# Single asset
python main.py --crypto btc
python main.py --crypto eth
python main.py --crypto bnb
```

---

## Results

### Dataset statistics

| Asset | Raw tweets | After filtering | Removed | Days |
|-------|-----------|----------------|---------|------|
| BTC   | 737 089   | **321 627**    | 56.4 %  | 36   |
| ETH   | 739 618   | **306 738**    | 58.5 %  | 36   |
| BNB   | 232 155   | **75 171**     | 67.6 %  | 36   |
| Total | 1 708 862 | **703 536**    | —       | —    |

Filtering steps: language=EN → deduplicate ID → deduplicate normalised text → remove retweets → remove spam/bots → min 4 words.

---

### Sentiment distribution — VADER vs FinBERT

| Asset | VADER pos | VADER neu | VADER neg | FinBERT pos | FinBERT neu | FinBERT neg | Agreement |
|-------|-----------|-----------|-----------|-------------|-------------|-------------|-----------|
| BTC   | 44.8 %    | 37.3 %    | 18.0 %    | 8.0 %       | 84.5 %      | 7.4 %       | 42.0 %    |
| ETH   | 47.3 %    | 37.0 %    | 15.7 %    | 6.5 %       | 87.9 %      | 5.6 %       | 40.7 %    |
| BNB   | 57.2 %    | 30.9 %    | 11.9 %    | 11.4 %      | 85.2 %      | 3.4 %       | 40.0 %    |

FinBERT classifies **84–88 % of tweets as neutral** due to domain shift (trained on formal financial text, applied to informal social media). VADER distributes more evenly and has significantly higher daily variance — making it preferable for correlation analysis at the daily aggregation level.

![VADER vs FinBERT comparison](code/results/cross_crypto/figures/vader_vs_finbert_crosscrypto.png)

---

### Pearson correlation — VADER sentiment vs daily return (lag 0)

| Asset | r      | p-value | Significant (α = 0.05)? | Post-hoc power |
|-------|--------|---------|------------------------|----------------|
| BTC   | 0.218  | 0.208   | No                     | 24.7 %         |
| ETH   | 0.259  | 0.133   | No                     | 33.1 %         |
| BNB   | 0.044  | 0.803   | No                     | 5.7 %          |

None of the lags 0–7 reaches p < 0.05 for any asset. BTC and ETH show a similar moderate positive pattern (r ≈ 0.22–0.26); BNB is near zero — consistent with its role as a utility token whose audience produces systematically positive but price-decoupled messages.

![Cross-asset correlations](code/results/cross_crypto/figures/cross_crypto_correlations.png)

---

### Statistical power analysis

At n = 36 observations, the minimum detectable correlation at 80 % power is **|r| ≈ 0.45**. The observed correlations for BTC and ETH fall well below this threshold — the null results reflect insufficient power, not necessarily the absence of an effect.

| r (hypothetical) | Power (n = 36) |
|-----------------|----------------|
| 0.10            | 8.5 %          |
| 0.15            | 12.5 %         |
| 0.20            | 18.7 %         |
| 0.25            | 27.0 %         |
| 0.30            | 37.5 %         |
| 0.45            | ~80 %          |

![Power curve](code/results/cross_crypto/figures/power_curve.png)

---

### Granger causality test (min p across lags 1–4)

|       | Raw series | Differenced series |
|-------|-----------|-------------------|
| BTC   | 0.610     | 0.480             |
| ETH   | 0.321     | 0.233             |
| BNB   | 0.211     | 0.211             |

No asset reaches p < 0.05 in either version. The differenced test is the methodologically correct form: BTC and ETH sentiment series are non-stationary (ADF p > 0.05), so first differencing is applied before testing.

---

### Return direction classifier (panel mode: BTC + ETH + BNB)

| Model                          | Accuracy | ROC-AUC |
|-------------------------------|----------|---------|
| DummyClassifier (most_frequent) | **0.563** | 0.500 |
| Logistic Regression            | 0.538    | 0.362   |
| Gradient Boosting              | 0.425    | 0.400   |

Neither ML model outperforms the naive baseline — consistent with the weak-form market efficiency hypothesis: a publicly available aggregated sentiment signal is quickly arbitraged away.

![Classifier results](code/results/cross_crypto/figures/prediction_results.png)

---

### Sample figures — BTC

| Sentiment vs Price | Lagged Correlations |
|---|---|
| ![](code/results/btc/figures/sentiment_vs_price.png) | ![](code/results/btc/figures/lagged_correlation.png) |

| Sentiment Distribution | Tweet Volume |
|---|---|
| ![](code/results/btc/figures/sentiment_distribution.png) | ![](code/results/btc/figures/tweet_volume.png) |

---

### FinBERT Architecture

`ProsusAI/finbert` — BERT-base fine-tuned on financial text for 3-class sentiment classification.

```
BertForSequenceClassification
├─ BertModel
│   ├─ BertEmbeddings        23 835 648 params
│   ├─ BertEncoder (× 12)    85 054 464 params
│   └─ BertPooler               590 592 params
├─ Dropout
└─ Linear (classifier)            2 307 params
─────────────────────────────────────────────
Total trainable: 109 484 547 params · 437.94 MB
```

Full `torchinfo` report: [`code/results/finbert_torchinfo.txt`](code/results/finbert_torchinfo.txt)

![FinBERT Netron graph](code/results/figures/netron_finbert_small.png)

---

## Testing

```bash
cd code

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src

# Skip FinBERT and full-dataset tests (fast, ~30 s)
pytest tests/ -v -m "not slow"
```

**354 tests** across 10 test files:

| File | Covers |
|------|--------|
| `test_data_loader.py` | CSV loading, date parsing, encoding |
| `test_preprocessor.py` | URL/mention removal, tokenisation, edge cases |
| `test_sentiment_analyzer.py` | VADER range, polarity, FinBERT (mock) |
| `test_price_fetcher.py` | BTC/ETH/BNB cache loading, fallback chain |
| `test_binance_loader.py` | 1-min OHLCV aggregation |
| `test_correlation_analyzer.py` | Pearson/Spearman, lags, Granger, ADF, differenced series |
| `test_power_analysis.py` | Fisher z power, min detectable r (vs. statistical tables) |
| `test_predictor.py` | No look-ahead bias, walk-forward splits, bootstrap CI, determinism |
| `test_data_integrity.py` | Cross-file consistency checks |
| `test_results_integrity.py` | Output CSV schema and value ranges |

---

## Methodology notes

- **Dual text format**: VADER receives the *original* tweet (capitalisation, punctuation, emoji are tonal signals); FinBERT receives *cleaned* text (URLs and mentions are noise for the transformer).
- **Granger test run twice**: once on raw series, once on first-differenced series after ADF confirms non-stationarity — the differenced version is the statistically valid one.
- **No look-ahead**: in the classifier, `features[t]` only use sentiment data from day `t`; the target is `sign(return[t+1])`. Walk-forward CV ensures test sets always follow training sets in time.
- **Reproducibility**: all random seeds fixed — `numpy.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`, `PYTHONHASHSEED=0`.

---

## CI / CD

GitHub Actions runs on every push / pull request:

- **Matrix**: Python 3.10 · 3.11 · 3.12
- **Steps**: install deps → download NLTK data → `pytest tests/ -v` → `flake8` lint

---

## License

MIT
