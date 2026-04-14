"""Main pipeline for Crypto Sentiment Analysis.

Usage:
    python main.py                          # Full pipeline (VADER + FinBERT)
    python main.py --skip-finbert           # VADER-only (fast, no GPU needed)
    python main.py --skip-finbert --skip-torchinfo  # Lightest run
    python main.py --sample 5000            # Subsample tweets for speed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    adf_test,
    correlation_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA_PROCESSED = Path("data/processed")


def parse_args():
    p = argparse.ArgumentParser(description="Crypto Sentiment Analysis Pipeline")
    p.add_argument("--tweets", default="data/raw/tweets.csv",
                   help="Path to tweets CSV")
    p.add_argument("--coin", default="bitcoin", help="CoinGecko coin ID")
    p.add_argument("--sample", type=int, default=0,
                   help="Subsample N tweets (0 = all)")
    p.add_argument("--skip-finbert", action="store_true",
                   help="Skip FinBERT (VADER only)")
    p.add_argument("--skip-torchinfo", action="store_true",
                   help="Skip torchinfo model summary")
    return p.parse_args()


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """Histogram of VADER compound scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["vader_compound"], bins=50, edgecolor="black", alpha=0.7,
            color="#4C72B0")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Tweet Sentiment (VADER Compound)")
    fig.tight_layout()
    fig.savefig(FIGURES / "sentiment_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved sentiment_distribution.png")


def plot_daily_sentiment_vs_price(daily: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Dual-axis time series: daily mean VADER vs BTC price."""
    merged = daily.merge(prices, on="date")
    dates = pd.to_datetime(merged["date"])

    fig, ax1 = plt.subplots(figsize=(14, 6))
    color_sent = "#4C72B0"
    color_price = "#DD8452"

    ax1.plot(dates, merged["mean_vader"], color=color_sent, linewidth=1.2,
             label="Mean VADER")
    ax1.fill_between(dates,
                     merged["mean_vader"] - merged["std_vader"],
                     merged["mean_vader"] + merged["std_vader"],
                     alpha=0.15, color=color_sent)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Mean VADER Compound", color=color_sent)
    ax1.tick_params(axis="y", labelcolor=color_sent)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax2 = ax1.twinx()
    ax2.plot(dates, merged["price"], color=color_price, linewidth=1.5,
             label="BTC Price (USD)")
    ax2.set_ylabel("BTC Price (USD)", color=color_price)
    ax2.tick_params(axis="y", labelcolor=color_price)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Daily Twitter Sentiment vs. Bitcoin Price")
    fig.tight_layout()
    fig.savefig(FIGURES / "sentiment_vs_price.png", dpi=150)
    plt.close(fig)
    logger.info("Saved sentiment_vs_price.png")


def plot_correlation_heatmap(corr_df: pd.DataFrame, title: str, fname: str) -> None:
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", fname)


def plot_lagged_correlation(lag_df: pd.DataFrame) -> None:
    """Bar chart of Pearson r at different lags."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lag_df["lag"], lag_df["pearson_r"], color="#4C72B0",
           edgecolor="black", alpha=0.8)
    for i, row in lag_df.iterrows():
        if row["pearson_p"] < 0.05:
            ax.text(row["lag"], row["pearson_r"] + 0.005, "*",
                    ha="center", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lagged Correlation: VADER Sentiment → BTC Return")
    ax.set_xticks(lag_df["lag"])
    fig.tight_layout()
    fig.savefig(FIGURES / "lagged_correlation.png", dpi=150)
    plt.close(fig)
    logger.info("Saved lagged_correlation.png")


def plot_tweet_volume(daily: pd.DataFrame) -> None:
    """Bar chart of daily tweet counts."""
    dates = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(dates, daily["tweet_count"], color="#55A868", alpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Tweet Count")
    ax.set_title("Daily Tweet Volume (BTC)")
    fig.tight_layout()
    fig.savefig(FIGURES / "tweet_volume.png", dpi=150)
    plt.close(fig)
    logger.info("Saved tweet_volume.png")


def plot_scatter_sentiment_return(merged: pd.DataFrame) -> None:
    """Scatter plot of daily mean VADER vs next-day return."""
    m = merged.dropna(subset=["mean_vader", "price_return"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m["mean_vader"], m["price_return"], alpha=0.6,
               edgecolors="k", linewidths=0.3, s=40)
    z = np.polyfit(m["mean_vader"], m["price_return"], 1)
    x_line = np.linspace(m["mean_vader"].min(), m["mean_vader"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1.5)
    ax.set_xlabel("Mean VADER Compound")
    ax.set_ylabel("BTC Daily Return")
    ax.set_title("Sentiment vs. BTC Return (same day)")
    fig.tight_layout()
    fig.savefig(FIGURES / "scatter_sentiment_return.png", dpi=150)
    plt.close(fig)
    logger.info("Saved scatter_sentiment_return.png")


def main():
    args = parse_args()

    for d in [FIGURES, TABLES, DATA_PROCESSED]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading tweet data")
    df = load_tweets(args.tweets)
    if args.sample > 0 and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d tweets", args.sample)
    logger.info("Dataset: %d tweets, date range %s to %s",
                len(df), df["date"].min(), df["date"].max())

    # ── Step 2: Preprocess ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing tweets")
    df = preprocess_dataframe(df)

    # ── Step 3: Sentiment Analysis ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Sentiment analysis")

    # VADER (on raw text)
    vader_scores = score_vader(df["full_text"])
    df = pd.concat([df, vader_scores], axis=1)
    logger.info("VADER complete. Mean compound: %.4f", df["vader_compound"].mean())

    # FinBERT (on cleaned text)
    if not args.skip_finbert:
        finbert_scores = score_finbert(df["cleaned_text"])
        df = pd.concat([df, finbert_scores], axis=1)
        logger.info(
            "FinBERT complete. Label distribution:\n%s",
            df["finbert_label"].value_counts().to_string()
        )

    # ── Step 4: torchinfo summary ──────────────────────────────────────
    if not args.skip_torchinfo and not args.skip_finbert:
        logger.info("=" * 60)
        logger.info("STEP 4: FinBERT model architecture (torchinfo)")
        from src.sentiment_analyzer import get_torchinfo_summary
        summary_str = get_torchinfo_summary()
        summary_path = RESULTS / "finbert_torchinfo_summary.txt"
        summary_path.write_text(summary_str)
        logger.info("torchinfo summary saved to %s", summary_path)
        print("\n" + summary_str + "\n")

    # ── Step 5: Fetch prices ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Fetching BTC price data")
    start_date = str(df["date"].min())
    end_date = str(df["date"].max())
    prices = load_or_fetch_prices(
        args.coin, start_date, end_date,
        cache_path=str(DATA_PROCESSED / f"{args.coin}_prices.csv"),
    )
    logger.info("Price data: %d days", len(prices))

    # ── Step 6: Aggregate & merge ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Aggregating daily sentiment & merging with prices")
    daily = aggregate_daily_sentiment(df)
    # Normalize date columns to string for reliable merge
    daily["date"] = daily["date"].astype(str)
    prices["date"] = prices["date"].astype(str)
    merged = daily.merge(prices, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    logger.info("Merged dataset: %d days", len(merged))

    merged.to_csv(DATA_PROCESSED / "merged_daily.csv", index=False)
    logger.info("Saved merged_daily.csv")

    # ── Step 7: Correlation analysis ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Correlation analysis")

    # Lagged correlations
    lag_df = compute_lagged_correlations(merged, "mean_vader", "price_return")
    lag_df.to_csv(TABLES / "lagged_correlations.csv", index=False)
    logger.info("Lagged correlations:\n%s", lag_df.to_string(index=False))

    # Granger causality
    granger_df = run_granger_test(merged, "mean_vader", "price_return")
    if not granger_df.empty:
        granger_df.to_csv(TABLES / "granger_causality.csv", index=False)
        logger.info("Granger causality:\n%s", granger_df.to_string(index=False))

    # ADF stationarity tests
    adf_results = []
    for col, name in [("mean_vader", "Daily Mean VADER"),
                      ("price_return", "BTC Daily Return")]:
        if col in merged.columns and merged[col].dropna().shape[0] > 10:
            adf_results.append(adf_test(merged[col], name))
    if adf_results:
        adf_df = pd.DataFrame(adf_results)
        adf_df.to_csv(TABLES / "adf_stationarity.csv", index=False)
        logger.info("ADF tests:\n%s", adf_df.to_string(index=False))

    # Correlation matrix
    corr_cols = ["mean_vader", "vader_pos_ratio", "vader_neg_ratio",
                 "tweet_count", "price_return", "log_return", "volume"]
    valid_cols = [c for c in corr_cols if c in merged.columns]
    corr = correlation_matrix(merged, valid_cols)
    corr.to_csv(TABLES / "correlation_matrix.csv")

    # ── Step 8: Visualisations ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Generating visualisations")

    plot_sentiment_distribution(df)
    plot_daily_sentiment_vs_price(daily, prices)
    plot_tweet_volume(daily)
    plot_scatter_sentiment_return(merged)
    plot_correlation_heatmap(
        corr, "Correlation Matrix (Pearson)", "correlation_heatmap.png"
    )
    if not lag_df.empty:
        plot_lagged_correlation(lag_df)

    # ── Step 9: Export processed data ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 9: Exporting processed data")
    export_cols = ["date", "full_text", "cleaned_text", "vader_compound",
                   "vader_pos", "vader_neg", "vader_neu"]
    if "finbert_label" in df.columns:
        export_cols += ["finbert_label", "finbert_score"]
    df[export_cols].to_csv(DATA_PROCESSED / "tweets_with_sentiment.csv", index=False)
    logger.info("Saved tweets_with_sentiment.csv")

    # ── Summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Tweets analysed: %d", len(df))
    logger.info("Date range: %s to %s", merged["date"].min(), merged["date"].max())
    logger.info("Mean VADER compound: %.4f", df["vader_compound"].mean())
    if "finbert_label" in df.columns:
        logger.info(
            "FinBERT label distribution:\n%s",
            df["finbert_label"].value_counts(normalize=True).to_string()
        )
    logger.info("Peak lag-0 Pearson r: %.4f (p=%.4f)",
                lag_df.iloc[0]["pearson_r"], lag_df.iloc[0]["pearson_p"])
    logger.info("Results in: %s", RESULTS.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
