"""Main pipeline for Crypto Sentiment Analysis.

Usage:
    python main.py                               # BTC, full pipeline (VADER + FinBERT)
    python main.py --crypto eth --skip-finbert   # ETH, VADER only
    python main.py --crypto all --skip-finbert   # All three cryptos, VADER only
    python main.py --sample 5000                 # Subsample tweets for speed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import seaborn as sns  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loader import load_tweets  # noqa: E402
from src.preprocessor import preprocess_dataframe  # noqa: E402
from src.sentiment_analyzer import score_vader, score_finbert  # noqa: E402
from src.price_fetcher import load_or_fetch_prices  # noqa: E402
from src.correlation_analyzer import (  # noqa: E402
    aggregate_daily_sentiment,
    compute_lagged_correlations,
    run_granger_test,
    run_granger_on_differenced,
    adf_test,
    correlation_matrix,
)
from src.power_analysis import correlation_power, min_detectable_r, power_table  # noqa: E402
from src.predictor import evaluate_all, evaluate_panel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS = Path("results")
DATA_PROCESSED = Path("data/processed")

CRYPTO_CONFIG: dict[str, dict] = {
    "btc": {
        "coin_id": "bitcoin",
        "tweets": "data/raw/tweets_btc.csv",
        "label": "Bitcoin (BTC)",
        "price_label": "Цена BTC (USD)",
    },
    "eth": {
        "coin_id": "ethereum",
        "tweets": "data/raw/tweets_eth.csv",
        "label": "Ethereum (ETH)",
        "price_label": "Цена ETH (USD)",
    },
    "bnb": {
        "coin_id": "binancecoin",
        "tweets": "data/raw/tweets_bnb.csv",
        "label": "Binance Coin (BNB)",
        "price_label": "Цена BNB (USD)",
    },
}

# Russian translations for model names (display only — CSV keeps English keys)
_MODEL_NAME_RU = {
    "Dummy (most_frequent)": "Dummy (наиболее частый класс)",
    "Dummy (stratified)":    "Dummy (стратифицированный)",
    "Logistic Regression":   "Логистическая регрессия",
    "Gradient Boosting":     "Градиентный бустинг",
}


def parse_args():
    p = argparse.ArgumentParser(description="Crypto Sentiment Analysis Pipeline")
    p.add_argument("--crypto", choices=["btc", "eth", "bnb", "all"], default="btc",
                   help="Cryptocurrency to analyse (default: btc). 'all' runs all three.")
    p.add_argument("--tweets", default=None,
                   help="Override tweets CSV path (ignored when --crypto all)")
    p.add_argument("--coin", default=None,
                   help="Override CoinGecko coin ID (ignored when --crypto all)")
    p.add_argument("--sample", type=int, default=0,
                   help="Subsample N tweets (0 = all)")
    p.add_argument("--skip-finbert", action="store_true",
                   help="Skip FinBERT (VADER only)")
    p.add_argument("--skip-torchinfo", action="store_true",
                   help="Skip torchinfo model summary")
    return p.parse_args()


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_sentiment_distribution(df: pd.DataFrame, figures: Path, crypto: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["vader_compound"], bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Оценка тональности VADER (compound)")
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение тональности твитов — {CRYPTO_CONFIG[crypto]['label']}")
    fig.tight_layout()
    _savefig(fig, figures / "sentiment_distribution.png")


def plot_daily_sentiment_vs_price(
    daily: pd.DataFrame,
    prices: pd.DataFrame,
    figures: Path,
    crypto: str,
) -> None:
    merged = daily.merge(prices, on="date")
    dates = pd.to_datetime(merged["date"])
    cfg = CRYPTO_CONFIG[crypto]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    c_sent, c_price = "#4C72B0", "#DD8452"

    ax1.plot(dates, merged["mean_vader"], color=c_sent, linewidth=1.2, label="Средняя тональность VADER")
    ax1.fill_between(
        dates,
        merged["mean_vader"] - merged["std_vader"],
        merged["mean_vader"] + merged["std_vader"],
        alpha=0.15, color=c_sent,
    )
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Средняя оценка VADER", color=c_sent)
    ax1.tick_params(axis="y", labelcolor=c_sent)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    ax2 = ax1.twinx()
    ax2.plot(dates, merged["price"], color=c_price, linewidth=1.5, label=cfg["price_label"])
    ax2.set_ylabel(cfg["price_label"], color=c_price)
    ax2.tick_params(axis="y", labelcolor=c_price)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"Дневная тональность твитов и цена {cfg['label']}")
    fig.tight_layout()
    _savefig(fig, figures / "sentiment_vs_price.png")


def plot_correlation_heatmap(
    corr_df: pd.DataFrame, title: str, figures: Path, fname: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, figures / fname)


def plot_lagged_correlation(
    lag_df: pd.DataFrame, figures: Path, crypto: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lag_df["lag"], lag_df["pearson_r"], color="#4C72B0", edgecolor="black", alpha=0.8)
    for _, row in lag_df.iterrows():
        if row["pearson_p"] < 0.05:
            ax.text(row["lag"], row["pearson_r"] + 0.005, "*",
                    ha="center", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Лаг (дни)")
    ax.set_ylabel("Коэффициент корреляции Пирсона (r)")
    ax.set_title(
        f"Лагированная корреляция: тональность VADER → доходность {CRYPTO_CONFIG[crypto]['label']}"
    )
    ax.set_xticks(lag_df["lag"])
    fig.tight_layout()
    _savefig(fig, figures / "lagged_correlation.png")


def plot_tweet_volume(daily: pd.DataFrame, figures: Path, crypto: str) -> None:
    dates = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(dates, daily["tweet_count"], color="#55A868", alpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_xlabel("Дата")
    ax.set_ylabel("Количество твитов")
    ax.set_title(f"Дневной объём твитов — {CRYPTO_CONFIG[crypto]['label']}")
    fig.tight_layout()
    _savefig(fig, figures / "tweet_volume.png")


def plot_scatter_sentiment_return(
    merged: pd.DataFrame, figures: Path, crypto: str
) -> None:
    m = merged.dropna(subset=["mean_vader", "price_return"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m["mean_vader"], m["price_return"], alpha=0.6,
               edgecolors="k", linewidths=0.3, s=40)
    if len(m) >= 2:
        z = np.polyfit(m["mean_vader"], m["price_return"], 1)
        x_line = np.linspace(m["mean_vader"].min(), m["mean_vader"].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)
    ax.set_xlabel("Средняя оценка тональности VADER")
    ax.set_ylabel(f"Дневная доходность {CRYPTO_CONFIG[crypto]['label']}")
    ax.set_title(f"Тональность vs. доходность {CRYPTO_CONFIG[crypto]['label']} (лаг 0)")
    fig.tight_layout()
    _savefig(fig, figures / "scatter_sentiment_return.png")


# ── Single-crypto pipeline ───────────────────────────────────────────────────

def run_single_crypto(crypto: str, args) -> Optional[dict]:
    """Run the full analysis pipeline for one cryptocurrency.

    Returns a summary dict for the cross-crypto comparison table, or None
    if the data file is missing.
    """
    cfg = CRYPTO_CONFIG[crypto]
    tweets_path = args.tweets if (args.tweets and args.crypto != "all") else cfg["tweets"]
    coin_id = args.coin if (args.coin and args.crypto != "all") else cfg["coin_id"]

    figures = RESULTS / crypto / "figures"
    tables = RESULTS / crypto / "tables"
    for d in [figures, tables, DATA_PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 1: Loading tweet data from %s", crypto.upper(), tweets_path)
    if not Path(tweets_path).exists():
        logger.error("[%s] Tweets file not found: %s — skipping", crypto.upper(), tweets_path)
        return None

    df = load_tweets(tweets_path)
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        logger.info("[%s] Sampled %d tweets", crypto.upper(), args.sample)
    logger.info("[%s] Dataset: %d tweets, date range %s to %s",
                crypto.upper(), len(df), df["date"].min(), df["date"].max())

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 2: Preprocessing (already cleaned by convert_data.py)", crypto.upper())
    df["cleaned_text"] = df["full_text"]

    # ── Step 3: Sentiment Analysis ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 3: Sentiment analysis", crypto.upper())

    vader_scores = score_vader(df["original_text"])
    df = pd.concat([df, vader_scores], axis=1)
    logger.info("[%s] VADER complete. Mean compound: %.4f",
                crypto.upper(), df["vader_compound"].mean())

    if not args.skip_finbert:
        finbert_scores = score_finbert(df["cleaned_text"])
        df = pd.concat([df, finbert_scores], axis=1)
        logger.info("[%s] FinBERT complete. Label distribution:\n%s",
                    crypto.upper(), df["finbert_label"].value_counts().to_string())

    # ── Step 4: torchinfo summary (BTC only to avoid repetition) ──────────
    if not args.skip_torchinfo and not args.skip_finbert and crypto == "btc":
        logger.info("=" * 60)
        logger.info("[%s] STEP 4: FinBERT model architecture (torchinfo)", crypto.upper())
        from src.sentiment_analyzer import get_torchinfo_summary
        summary_str = get_torchinfo_summary()
        summary_path = RESULTS / "finbert_torchinfo_summary.txt"
        summary_path.write_text(summary_str)
        logger.info("torchinfo summary saved to %s", summary_path)

    # ── Step 5: Fetch prices ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 5: Fetching price data for %s", crypto.upper(), coin_id)
    start_date = str(df["date"].min())
    end_date = str(df["date"].max())
    prices = load_or_fetch_prices(
        coin_id, start_date, end_date,
        cache_path=str(DATA_PROCESSED / f"{coin_id}_prices.csv"),
    )
    logger.info("[%s] Price data: %d days", crypto.upper(), len(prices))

    # ── Step 6: Aggregate & merge ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 6: Aggregating daily sentiment & merging with prices", crypto.upper())
    daily = aggregate_daily_sentiment(df)
    daily["date"] = daily["date"].astype(str)
    prices["date"] = prices["date"].astype(str)
    merged = daily.merge(prices, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    logger.info("[%s] Merged dataset: %d days", crypto.upper(), len(merged))

    merged.to_csv(DATA_PROCESSED / f"merged_daily_{crypto}.csv", index=False)

    # ── Step 7: Correlation analysis ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 7: Correlation analysis", crypto.upper())

    lag_df = compute_lagged_correlations(merged, "mean_vader", "price_return")
    lag_df.to_csv(tables / "lagged_correlations.csv", index=False)
    logger.info("[%s] Lagged correlations:\n%s", crypto.upper(), lag_df.to_string(index=False))

    granger_df = run_granger_test(merged, "mean_vader", "price_return")
    if not granger_df.empty:
        granger_df.to_csv(tables / "granger_causality.csv", index=False)
        logger.info("[%s] Granger causality:\n%s", crypto.upper(), granger_df.to_string(index=False))

    adf_results = []
    for col, name in [("mean_vader", "Daily Mean VADER"),
                      ("price_return", f"{crypto.upper()} Daily Return")]:
        if col in merged.columns and merged[col].dropna().shape[0] > 10:
            adf_results.append(adf_test(merged[col], name))
    adf_df = pd.DataFrame()
    if adf_results:
        adf_df = pd.DataFrame(adf_results)
        adf_df.to_csv(tables / "adf_stationarity.csv", index=False)

    corr_cols = ["mean_vader", "vader_pos_ratio", "vader_neg_ratio",
                 "tweet_count", "price_return", "log_return", "volume"]
    valid_cols = [c for c in corr_cols if c in merged.columns]
    corr = correlation_matrix(merged, valid_cols)
    corr.to_csv(tables / "correlation_matrix.csv")

    # ── Step 8: Visualisations ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 8: Generating visualisations", crypto.upper())

    plot_sentiment_distribution(df, figures, crypto)
    plot_daily_sentiment_vs_price(daily, prices, figures, crypto)
    plot_tweet_volume(daily, figures, crypto)
    plot_scatter_sentiment_return(merged, figures, crypto)
    plot_correlation_heatmap(
        corr, f"Матрица корреляций — {CRYPTO_CONFIG[crypto]['label']}",
        figures, "correlation_heatmap.png",
    )
    if not lag_df.empty:
        plot_lagged_correlation(lag_df, figures, crypto)

    # ── Step 8b: VADER vs FinBERT comparison ──────────────────────────────
    if "finbert_label" in df.columns:
        df["vader_label"] = df["vader_compound"].apply(
            lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
        )
        comparison = pd.DataFrame({
            "metric": ["positive_ratio", "negative_ratio", "neutral_ratio", "mean_confidence"],
            "VADER": [
                (df["vader_label"] == "positive").mean(),
                (df["vader_label"] == "negative").mean(),
                (df["vader_label"] == "neutral").mean(),
                df["vader_compound"].abs().mean(),
            ],
            "FinBERT": [
                (df["finbert_label"] == "positive").mean(),
                (df["finbert_label"] == "negative").mean(),
                (df["finbert_label"] == "neutral").mean(),
                df["finbert_score"].mean(),
            ],
        })
        comparison.to_csv(tables / "vader_vs_finbert_comparison.csv", index=False)
        crosstab = pd.crosstab(df["vader_label"], df["finbert_label"],
                               margins=True, margins_name="Total")
        crosstab.to_csv(tables / "vader_finbert_crosstab.csv")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        labels_order = ["positive", "negative", "neutral"]
        labels_ru = ["позитивная", "негативная", "нейтральная"]
        colors = ["#55A868", "#C44E52", "#8C8C8C"]
        vader_counts = df["vader_label"].value_counts(normalize=True)
        finbert_counts = df["finbert_label"].value_counts(normalize=True)
        for ax, counts, title in [
            (axes[0], vader_counts, "VADER"),
            (axes[1], finbert_counts, "FinBERT"),
        ]:
            vals = [counts.get(l, 0) for l in labels_order]
            ax.bar(labels_ru, vals, color=colors, edgecolor="black", alpha=0.8)
            ax.set_ylabel("Доля твитов")
            ax.set_title(f"{title} — {CRYPTO_CONFIG[crypto]['label']}")
            ax.set_ylim(0, 1)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
        fig.suptitle("Распределение тональности: VADER vs. FinBERT", fontsize=12, y=1.01)
        fig.tight_layout()
        _savefig(fig, figures / "vader_vs_finbert_comparison.png")

    # ── Step 9: Export processed data ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 9: Exporting processed data", crypto.upper())
    export_cols = ["date", "original_text", "full_text", "cleaned_text",
                   "vader_compound", "vader_pos", "vader_neg", "vader_neu"]
    if "finbert_label" in df.columns:
        export_cols += ["finbert_label", "finbert_score"]
    df[export_cols].to_csv(DATA_PROCESSED / f"tweets_with_sentiment_{crypto}.csv", index=False)

    # ── Step 10: Granger on differenced series ─────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 10: Granger causality on (possibly differenced) series", crypto.upper())
    granger_diff_df = run_granger_on_differenced(merged, "mean_vader", "price_return")
    if not granger_diff_df.empty:
        granger_diff_df.to_csv(tables / "granger_differenced.csv", index=False)
        logger.info("[%s] Granger (differenced):\n%s",
                    crypto.upper(), granger_diff_df.to_string(index=False))

    # ── Step 11: Power analysis ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 11: Post-hoc power analysis", crypto.upper())
    n_days = len(merged)
    _lag0_rows = lag_df[lag_df["lag"] == 0] if not lag_df.empty else pd.DataFrame()
    r_observed = float(_lag0_rows["pearson_r"].iloc[0]) if not _lag0_rows.empty else 0.0
    observed_power = correlation_power(r_observed, n_days)
    mdr = min_detectable_r(n_days, power=0.8)
    logger.info(
        "[%s] n=%d, observed r=%.4f, power=%.4f, MDR@0.8=%.4f",
        crypto.upper(), n_days, r_observed, observed_power, mdr,
    )
    pow_obs = pd.DataFrame([{
        "crypto": crypto,
        "n": n_days,
        "observed_r": round(r_observed, 4),
        "observed_power": round(observed_power, 4),
        "min_detectable_r_at_0.8": round(mdr, 4),
    }])
    pow_obs.to_csv(tables / "power_for_observed.csv", index=False)

    # ── Step 12: Baseline direction-of-return classifier ───────────────────
    logger.info("=" * 60)
    logger.info("[%s] STEP 12: Baseline direction-of-return classifier", crypto.upper())
    pred_df = evaluate_all(merged, crypto_label=crypto)
    if not pred_df.empty:
        pred_path = RESULTS / "prediction"
        pred_path.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pred_path / f"prediction_results_{crypto}.csv", index=False)
        logger.info("[%s] Prediction results:\n%s",
                    crypto.upper(), pred_df[["model", "accuracy", "f1", "roc_auc"]].to_string(index=False))

    # ── Build summary for comparison table ────────────────────────────────
    lag0 = lag_df[lag_df["lag"] == 0].iloc[0] if not lag_df.empty else {}
    granger_min_p = granger_df["p_value"].min() if not granger_df.empty else float("nan")
    adf_sent_p = float("nan")
    adf_ret_p = float("nan")
    if not adf_df.empty:
        sent_row = adf_df[adf_df["series"] == "Daily Mean VADER"]
        ret_row = adf_df[adf_df["series"] == f"{crypto.upper()} Daily Return"]
        if not sent_row.empty:
            adf_sent_p = float(sent_row["p_value"].iloc[0])
        if not ret_row.empty:
            adf_ret_p = float(ret_row["p_value"].iloc[0])

    summary = {
        "crypto": crypto,
        "n_tweets": len(df),
        "n_days": len(merged),
        "mean_sentiment": round(float(df["vader_compound"].mean()), 4),
        "std_sentiment": round(float(df["vader_compound"].std()), 4),
        "corr_r": round(float(lag0.get("pearson_r", float("nan"))), 4),
        "corr_p": round(float(lag0.get("pearson_p", float("nan"))), 4),
        "granger_min_p": round(float(granger_min_p), 4),
        "adf_sentiment_p": round(float(adf_sent_p), 4),
        "adf_return_p": round(float(adf_ret_p), 4),
        "price_min": round(float(prices["price"].min()), 2),
        "price_max": round(float(prices["price"].max()), 2),
    }

    logger.info("=" * 60)
    logger.info("[%s] PIPELINE COMPLETE — %d tweets, %d days, mean VADER %.4f",
                crypto.upper(), len(df), len(merged), df["vader_compound"].mean())
    logger.info("=" * 60)

    return summary


# ── Cross-crypto comparison ──────────────────────────────────────────────────

def generate_comparison_plot(comparison: pd.DataFrame) -> None:
    """Bar chart comparing Pearson r across the three cryptos."""
    out_dir = RESULTS / "cross_crypto"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(comparison["crypto"].str.upper(), comparison["corr_r"],
                  color=colors[:len(comparison)], edgecolor="black", alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Криптовалюта")
    ax.set_ylabel("Коэффициент корреляции Пирсона (r, лаг 0)")
    ax.set_title("Корреляция тональности и доходности по криптовалютам")
    for bar, val in zip(bars, comparison["corr_r"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.005 if val >= 0 else -0.015),
                f"{val:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    path = out_dir / "correlation_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Sprint-2 visualisations ──────────────────────────────────────────────────

def plot_power_curve(figures: Path) -> None:
    """Power vs. r for n=36 (Fisher z-transformation)."""
    figures.mkdir(parents=True, exist_ok=True)
    r_range = np.linspace(0.01, 0.99, 200)
    powers = [correlation_power(r, 36) for r in r_range]

    mdr = min_detectable_r(36, power=0.8)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r_range, powers, color="#4C72B0", linewidth=2)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="Мощность 80%")
    ax.axvline(mdr, color="gray", linestyle=":", linewidth=1.2,
               label=f"Минимально детектируемая r = {mdr:.2f}")
    for r_obs, lbl, clr in [(0.218, "BTC (r = 0.218)", "#4C72B0"),
                             (0.259, "ETH (r = 0.259)", "#DD8452"),
                             (0.044, "BNB (r = 0.044)", "#55A868")]:
        ax.axvline(abs(r_obs), color=clr, linestyle="-.", linewidth=1.2, label=lbl)
    ax.set_xlabel("|r| (коэффициент корреляции Пирсона)")
    ax.set_ylabel("Статистическая мощность")
    ax.set_title("Апостериорный анализ мощности: n = 36, α = 0.05")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _savefig(fig, figures / "power_curve.png")


def plot_cross_crypto_correlations(comparison: pd.DataFrame, figures: Path) -> None:
    """Grouped bar chart: correlation at lag 0 for BTC, ETH, BNB."""
    figures.mkdir(parents=True, exist_ok=True)
    if comparison.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(
        comparison["crypto"].str.upper(),
        comparison["corr_r"],
        color=colors[:len(comparison)],
        edgecolor="black", alpha=0.85,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Криптовалюта")
    ax.set_ylabel("Коэффициент корреляции Пирсона (r)")
    ax.set_title("Корреляция тональности и доходности по криптовалютам (лаг 0)")
    for bar, val in zip(bars, comparison["corr_r"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.005 if val >= 0 else -0.018),
            f"{val:.3f}", ha="center", fontsize=11,
        )
    fig.tight_layout()
    _savefig(fig, figures / "cross_crypto_correlations.png")


def plot_prediction_results(pred_df: pd.DataFrame, figures: Path) -> None:
    """Bar chart of accuracy with CI for each model and crypto.

    Panel group is rendered with hatching and a distinct colour to separate it
    visually from the per-crypto bars.
    """
    figures.mkdir(parents=True, exist_ok=True)
    if pred_df.empty or "accuracy" not in pred_df.columns:
        return

    model_names = list(pred_df["model"].unique())
    # Put panel last so it stays visually grouped
    all_cryptos = [c for c in pred_df["crypto"].unique() if c != "panel"]
    if "panel" in pred_df["crypto"].values:
        all_cryptos.append("panel")

    n_groups = len(all_cryptos)
    n_models = len(model_names)
    width = 0.8 / max(n_groups, 1)

    # Colors for per-crypto bars; panel gets a dedicated dark color + hatching
    per_crypto_colors = {"btc": "#4C72B0", "eth": "#DD8452", "bnb": "#55A868"}
    panel_color = "#2D2D2D"

    fig, ax = plt.subplots(figsize=(max(11, n_models * 2.8), 5))
    x = np.arange(n_models)

    for i, crypto in enumerate(all_cryptos):
        sub = pred_df[pred_df["crypto"] == crypto].set_index("model")
        vals = [float(sub.loc[m, "accuracy"]) if m in sub.index else float("nan")
                for m in model_names]
        lo   = [float(sub.loc[m, "accuracy_lower"]) if m in sub.index else float("nan")
                for m in model_names]
        hi   = [float(sub.loc[m, "accuracy_upper"]) if m in sub.index else float("nan")
                for m in model_names]
        err_lo = [v - l if not np.isnan(v) else 0 for v, l in zip(vals, lo)]
        err_hi = [h - v if not np.isnan(v) else 0 for v, h in zip(vals, hi)]
        offset = (i - n_groups / 2 + 0.5) * width

        is_panel = crypto == "panel"
        color = panel_color if is_panel else per_crypto_colors.get(crypto, "#8172B2")
        hatch = "///" if is_panel else None
        label = "Панель (BTC+ETH+BNB)" if is_panel else crypto.upper()

        ax.bar(x + offset, vals, width, label=label,
               color=color, alpha=0.85, edgecolor="black",
               hatch=hatch, linewidth=0.8)
        ax.errorbar(x + offset, vals,
                    yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=4, linewidth=1)

    # Vertical separator before panel group
    if "panel" in all_cryptos and len(all_cryptos) > 1:
        panel_idx = all_cryptos.index("panel")
        sep_x = x + (panel_idx - n_groups / 2) * width
        for sx in sep_x:
            ax.axvline(sx - width * 0.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Случайный baseline (50%)")
    ax.set_xticks(x)
    xticklabels_ru = [_MODEL_NAME_RU.get(m, m) for m in model_names]
    ax.set_xticklabels(xticklabels_ru, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (с 95% доверительным интервалом)")
    ax.set_title("Классификация направления дневной доходности: walk-forward CV")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=min(n_groups + 1, 3))
    fig.tight_layout()
    _savefig(fig, figures / "prediction_results.png")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cryptos = ["btc", "eth", "bnb"] if args.crypto == "all" else [args.crypto]
    summaries = []

    for crypto in cryptos:
        summary = run_single_crypto(crypto, args)
        if summary:
            summaries.append(summary)

    if args.crypto == "all" and summaries:
        cross_dir = RESULTS / "cross_crypto"
        cross_dir.mkdir(parents=True, exist_ok=True)
        comparison = pd.DataFrame(summaries)
        comparison.to_csv(cross_dir / "comparison.csv", index=False)
        logger.info("Cross-crypto comparison saved to %s", cross_dir / "comparison.csv")
        logger.info("\n%s", comparison.to_string(index=False))
        generate_comparison_plot(comparison)
        plot_cross_crypto_correlations(comparison, cross_dir / "figures")

        # Cross-crypto power analysis table
        n_ref = 36
        r_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
        pow_df = power_table([n_ref], r_vals)
        pow_df.to_csv(cross_dir / "power_analysis.csv", index=False)
        logger.info("Power analysis table saved.")

        # Aggregate per-crypto power_for_observed
        power_rows = []
        for crypto in ["btc", "eth", "bnb"]:
            p = RESULTS / crypto / "tables" / "power_for_observed.csv"
            if p.exists():
                power_rows.append(pd.read_csv(p))
        if power_rows:
            pd.concat(power_rows).to_csv(cross_dir / "power_for_observed.csv", index=False)

        # Aggregate per-crypto prediction results
        pred_rows = []
        for crypto in ["btc", "eth", "bnb"]:
            p = RESULTS / "prediction" / f"prediction_results_{crypto}.csv"
            if p.exists():
                pred_rows.append(pd.read_csv(p))

        # Panel evaluation across all three cryptos
        panel_dfs = {}
        for crypto in ["btc", "eth", "bnb"]:
            p = DATA_PROCESSED / f"merged_daily_{crypto}.csv"
            if p.exists():
                panel_dfs[crypto] = pd.read_csv(p)

        panel_results = pd.DataFrame()
        if len(panel_dfs) > 1:
            logger.info("Running panel evaluation (%d cryptos combined)…", len(panel_dfs))
            panel_results = evaluate_panel(panel_dfs)
            if not panel_results.empty:
                panel_results.to_csv(
                    RESULTS / "prediction" / "prediction_results_panel.csv", index=False
                )
                logger.info(
                    "Panel prediction results:\n%s",
                    panel_results[["model", "accuracy", "f1", "roc_auc"]].to_string(index=False),
                )

        pred_all = pd.DataFrame()
        if pred_rows or not panel_results.empty:
            pred_all = pd.concat(
                pred_rows + ([panel_results] if not panel_results.empty else []),
                ignore_index=True,
            )
            pred_all.to_csv(RESULTS / "prediction" / "prediction_results.csv", index=False)
            logger.info("Aggregated prediction_results.csv saved.")

        plot_power_curve(cross_dir / "figures")
        plot_prediction_results(pred_all, cross_dir / "figures")


if __name__ == "__main__":
    main()
