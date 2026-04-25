#!/usr/bin/env python3
"""
FinBERT full-corpus inference with checkpointing.

Processes BTC, ETH, BNB (skips cryptos already done).
Saves checkpoint every CHECKPOINT_EVERY rows — safe to kill and restart.

Outputs after inference:
  data/processed/tweets_with_sentiment_{crypto}.csv   — finbert_label + finbert_score added
  data/processed/finbert_checkpoint_{crypto}.csv      — resume file (deleted on completion)

  results/{crypto}/tables/vader_vs_finbert_comparison.csv
  results/{crypto}/tables/vader_finbert_crosstab.csv
  results/{crypto}/figures/vader_vs_finbert_comparison.png

  results/cross_crypto/vader_finbert_summary.csv
  results/cross_crypto/figures/vader_vs_finbert_crosscrypto.png
  results/cross_crypto/comparison.csv                 — finbert columns added

  results/finbert_timing.txt                          — actual run times
  finbert_run.log                                     — full log of this run

Usage:
    python run_finbert_full.py [--crypto btc eth bnb]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "finbert_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_PROCESSED = BASE / "data" / "processed"
RESULTS = BASE / "results"

# ── Constants ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
CHECKPOINT_EVERY = 10_000   # rows between checkpoint saves

# ── FinBERT pipeline (lazy singleton) ─────────────────────────────────────────

_pipe = None


def _get_pipe():
    global _pipe
    if _pipe is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT (ProsusAI/finbert) — CPU-only...")
        _pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
        logger.info("FinBERT loaded.")
    return _pipe


# ── Inference with checkpointing ──────────────────────────────────────────────

def finbert_infer_with_checkpoint(texts: pd.Series, crypto: str) -> pd.DataFrame:
    """Run FinBERT on `texts` with checkpointing.

    Args:
        texts: Series of cleaned tweet strings with 0-based integer index.
        crypto: Lowercase name used for checkpoint filename.

    Returns:
        DataFrame(finbert_label, finbert_score) with same 0-based row count as texts.
    """
    checkpoint_path = DATA_PROCESSED / f"finbert_checkpoint_{crypto}.csv"
    pipe = _get_pipe()
    total = len(texts)

    # Load or initialise checkpoint
    if checkpoint_path.exists():
        ckpt_df = pd.read_csv(checkpoint_path)
        start_row = len(ckpt_df)
        logger.info(
            "[%s] Checkpoint: %d / %d rows already done", crypto.upper(), start_row, total
        )
    else:
        ckpt_df = pd.DataFrame(columns=["row_idx", "finbert_label", "finbert_score"])
        start_row = 0
        logger.info("[%s] No checkpoint — starting from row 0 (total %d)", crypto.upper(), total)

    if start_row >= total:
        logger.info("[%s] Already fully processed.", crypto.upper())
        return _ckpt_to_df(ckpt_df)

    text_list = texts.tolist()
    remaining = text_list[start_row:]
    new_rows: list[dict] = []
    rows_since_ckpt = 0

    try:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start: batch_start + BATCH_SIZE]
            batch = [str(t) if str(t).strip() else "neutral" for t in batch]
            out = pipe(batch)

            for j, r in enumerate(out):
                new_rows.append(
                    {
                        "row_idx": start_row + batch_start + j,
                        "finbert_label": r["label"].lower(),
                        "finbert_score": r["score"],
                    }
                )

            rows_since_ckpt += len(out)

            # Periodic checkpoint
            if rows_since_ckpt >= CHECKPOINT_EVERY:
                _write_checkpoint(ckpt_df, new_rows, checkpoint_path)
                done = start_row + len(new_rows)
                logger.info(
                    "[%s] Checkpoint saved: %d / %d  (%.1f%%)",
                    crypto.upper(), done, total, 100.0 * done / total,
                )
                rows_since_ckpt = 0

            # Log progress every 50K rows
            processed_now = len(new_rows)
            if processed_now > 0 and (processed_now % 50_000) < BATCH_SIZE:
                done = start_row + processed_now
                logger.info(
                    "[%s] Progress: %d / %d  (%.1f%%)",
                    crypto.upper(), done, total, 100.0 * done / total,
                )

    except KeyboardInterrupt:
        logger.info("[%s] Interrupted — saving checkpoint before exit...", crypto.upper())
        _write_checkpoint(ckpt_df, new_rows, checkpoint_path)
        done = start_row + len(new_rows)
        logger.info("[%s] Checkpoint saved at row %d. Re-run to resume.", crypto.upper(), done)
        raise

    # Final checkpoint save
    _write_checkpoint(ckpt_df, new_rows, checkpoint_path)
    final_df = pd.read_csv(checkpoint_path)
    logger.info("[%s] Inference complete: %d rows total.", crypto.upper(), len(final_df))
    return _ckpt_to_df(final_df)


def _write_checkpoint(
    existing: pd.DataFrame, new_rows: list[dict], path: Path
) -> None:
    if not new_rows:
        return
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(path, index=False)


def _ckpt_to_df(ckpt: pd.DataFrame) -> pd.DataFrame:
    ckpt = ckpt.sort_values("row_idx").reset_index(drop=True)
    return pd.DataFrame(
        {"finbert_label": ckpt["finbert_label"].values, "finbert_score": ckpt["finbert_score"].values}
    )


# ── Per-crypto: comparison tables & plot ──────────────────────────────────────

_LABELS_ORDER = ["positive", "negative", "neutral"]
_LABELS_RU    = ["позитивная", "негативная", "нейтральная"]
_COLORS       = ["#55A868", "#C44E52", "#8C8C8C"]


def _vader_label(compound: float) -> str:
    if compound > 0.05:
        return "positive"
    if compound < -0.05:
        return "negative"
    return "neutral"


def generate_comparison_outputs(df: pd.DataFrame, crypto: str) -> dict:
    """Generate tables + plot for one crypto. Returns a summary dict."""
    tables_dir = RESULTS / crypto / "tables"
    figures_dir = RESULTS / crypto / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["vader_label"] = df["vader_compound"].apply(_vader_label)

    agree = (df["vader_label"] == df["finbert_label"]).mean()

    # Comparison table
    comp = pd.DataFrame(
        {
            "metric": [
                "positive_ratio", "negative_ratio", "neutral_ratio",
                "agreement", "mean_confidence",
            ],
            "VADER": [
                (df["vader_label"] == "positive").mean(),
                (df["vader_label"] == "negative").mean(),
                (df["vader_label"] == "neutral").mean(),
                agree,
                df["vader_compound"].abs().mean(),
            ],
            "FinBERT": [
                (df["finbert_label"] == "positive").mean(),
                (df["finbert_label"] == "negative").mean(),
                (df["finbert_label"] == "neutral").mean(),
                agree,
                df["finbert_score"].mean(),
            ],
        }
    )
    comp.to_csv(tables_dir / "vader_vs_finbert_comparison.csv", index=False)

    # Crosstab
    crosstab = pd.crosstab(
        df["vader_label"], df["finbert_label"],
        margins=True, margins_name="Итого",
    )
    crosstab.to_csv(tables_dir / "vader_finbert_crosstab.csv")

    # Plot
    vader_cnts   = df["vader_label"].value_counts(normalize=True)
    finbert_cnts = df["finbert_label"].value_counts(normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cnts, title in [
        (axes[0], vader_cnts, "VADER"),
        (axes[1], finbert_cnts, "FinBERT"),
    ]:
        vals = [cnts.get(lb, 0.0) for lb in _LABELS_ORDER]
        ax.bar(_LABELS_RU, vals, color=_COLORS, edgecolor="black", alpha=0.8)
        ax.set_ylabel("Доля твитов")
        ax.set_title(f"{title} — {crypto.upper()}")
        ax.set_ylim(0, 1)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)

    fig.suptitle(
        f"Распределение тональности: VADER vs. FinBERT — {crypto.upper()}\n"
        f"Согласованность классификаций: {agree:.1%}",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "vader_vs_finbert_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "[%s] agree=%.1f%%  VADER pos/neu/neg=%.1f%%/%.1f%%/%.1f%%"
        "  FinBERT pos/neu/neg=%.1f%%/%.1f%%/%.1f%%",
        crypto.upper(), agree * 100,
        (df["vader_label"] == "positive").mean() * 100,
        (df["vader_label"] == "neutral").mean() * 100,
        (df["vader_label"] == "negative").mean() * 100,
        (df["finbert_label"] == "positive").mean() * 100,
        (df["finbert_label"] == "neutral").mean() * 100,
        (df["finbert_label"] == "negative").mean() * 100,
    )

    return {
        "crypto": crypto.upper(),
        "vader_pos_pct":    round((df["vader_label"] == "positive").mean() * 100, 2),
        "vader_neu_pct":    round((df["vader_label"] == "neutral").mean()  * 100, 2),
        "vader_neg_pct":    round((df["vader_label"] == "negative").mean() * 100, 2),
        "finbert_pos_pct":  round((df["finbert_label"] == "positive").mean() * 100, 2),
        "finbert_neu_pct":  round((df["finbert_label"] == "neutral").mean()  * 100, 2),
        "finbert_neg_pct":  round((df["finbert_label"] == "negative").mean() * 100, 2),
        "agreement_pct":    round(agree * 100, 2),
        "n":                len(df),
    }


# ── Cross-crypto summary ───────────────────────────────────────────────────────

def generate_cross_crypto_summary(summaries: list[dict]) -> None:
    cross_dir = RESULTS / "cross_crypto"
    figures_dir = cross_dir / "figures"
    cross_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Weighted panel row
    total_n = sum(s["n"] for s in summaries)
    panel = {
        "crypto": "Panel",
        "vader_pos_pct":   round(sum(s["vader_pos_pct"]   * s["n"] for s in summaries) / total_n, 2),
        "vader_neu_pct":   round(sum(s["vader_neu_pct"]   * s["n"] for s in summaries) / total_n, 2),
        "vader_neg_pct":   round(sum(s["vader_neg_pct"]   * s["n"] for s in summaries) / total_n, 2),
        "finbert_pos_pct": round(sum(s["finbert_pos_pct"] * s["n"] for s in summaries) / total_n, 2),
        "finbert_neu_pct": round(sum(s["finbert_neu_pct"] * s["n"] for s in summaries) / total_n, 2),
        "finbert_neg_pct": round(sum(s["finbert_neg_pct"] * s["n"] for s in summaries) / total_n, 2),
        "agreement_pct":   round(sum(s["agreement_pct"]   * s["n"] for s in summaries) / total_n, 2),
        "n": total_n,
    }

    all_rows = summaries + [panel]
    summary_df = pd.DataFrame(all_rows)
    summary_df.columns = [
        "Криптовалюта", "VADER pos %", "VADER neu %", "VADER neg %",
        "FinBERT pos %", "FinBERT neu %", "FinBERT neg %", "Согласие, %", "N",
    ]
    summary_df.to_csv(cross_dir / "vader_finbert_summary.csv", index=False)
    logger.info("Saved cross_crypto/vader_finbert_summary.csv")

    # Cross-crypto plot: 3 groups (BTC/ETH/BNB), 6 bars each (VADER+FinBERT × pos/neu/neg)
    crypto_labels = [s["crypto"] for s in summaries]
    x = np.arange(len(crypto_labels))
    width = 0.13

    # (attr_key, ru_label, vader_color, finbert_color)
    dims = [
        ("pos", "позитивная", "#55A868", "#2D8044"),
        ("neu", "нейтральная", "#8C8C8C", "#484848"),
        ("neg", "негативная", "#C44E52", "#7A1F22"),
    ]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (key, lbl, cv, cf) in enumerate(dims):
        vader_vals   = [s[f"vader_{key}_pct"]   / 100 for s in summaries]
        finbert_vals = [s[f"finbert_{key}_pct"] / 100 for s in summaries]
        offset_v = (2 * i - len(dims) + 0.5) * width
        offset_f = offset_v + width

        ax.bar(x + offset_v, vader_vals, width, label=f"VADER {lbl}",
               color=cv, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.bar(x + offset_f, finbert_vals, width, label=f"FinBERT {lbl}",
               color=cf, alpha=0.85, edgecolor="black", linewidth=0.5, hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(crypto_labels, fontsize=13)
    ax.set_ylabel("Доля твитов")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "VADER vs. FinBERT: распределение тональности по криптовалютам\n"
        "(сплошная заливка — VADER, косая штриховка — FinBERT)"
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "vader_vs_finbert_crosscrypto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cross_crypto/figures/vader_vs_finbert_crosscrypto.png")


def update_comparison_csv(summaries: list[dict]) -> None:
    """Add FinBERT ratio columns to cross_crypto/comparison.csv."""
    comp_path = RESULTS / "cross_crypto" / "comparison.csv"
    if not comp_path.exists():
        logger.warning("comparison.csv not found — skipping update")
        return

    comp = pd.read_csv(comp_path)
    for s in summaries:
        mask = comp["crypto"] == s["crypto"].lower()
        if mask.sum() == 0:
            continue
        comp.loc[mask, "finbert_pos_ratio"]      = round(s["finbert_pos_pct"] / 100, 4)
        comp.loc[mask, "finbert_neu_ratio"]      = round(s["finbert_neu_pct"] / 100, 4)
        comp.loc[mask, "finbert_neg_ratio"]      = round(s["finbert_neg_pct"] / 100, 4)
        comp.loc[mask, "vader_finbert_agreement"] = round(s["agreement_pct"] / 100, 4)

    comp.to_csv(comp_path, index=False)
    logger.info("Updated cross_crypto/comparison.csv with FinBERT columns")


# ── Timing log ─────────────────────────────────────────────────────────────────

def update_timing_log(timings: dict[str, float | None]) -> None:
    lines = [
        "FinBERT inference timing — CPU-only, batch_size=16",
        "====================================================",
        "Калибровочный прогон (500 сообщений): 40.27 сек, 12.42 сообщений/сек",
        "",
        "Фактическое время полного прогона:",
    ]
    measured = []
    for crypto in ["btc", "eth", "bnb"]:
        t = timings.get(crypto)
        if t is None:
            lines.append(f"  {crypto.upper()}: не замерено (прогон не выполнялся в данной сессии)")
        else:
            lines.append(f"  {crypto.upper()}: {t:.0f} сек ({t / 3600:.2f} ч)")
            measured.append(t)
    if measured:
        total_sec = sum(measured)
        lines.append(f"  Итого (измеренное): {total_sec:.0f} сек ({total_sec / 3600:.2f} ч)")

    timing_path = RESULTS / "finbert_timing.txt"
    timing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated finbert_timing.txt")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crypto", nargs="+", default=["btc", "eth", "bnb"],
        help="Cryptos to process (default: btc eth bnb)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FinBERT full-corpus inference started")
    logger.info("Requested: %s", args.crypto)
    logger.info("=" * 70)

    # Determine which cryptos still need FinBERT
    to_process: list[str] = []
    for crypto in args.crypto:
        sent_path = DATA_PROCESSED / f"tweets_with_sentiment_{crypto}.csv"
        if not sent_path.exists():
            logger.warning("[%s] tweets_with_sentiment file not found — skip", crypto.upper())
            continue
        header = pd.read_csv(sent_path, nrows=0)
        if "finbert_label" in header.columns:
            logger.info("[%s] Already has FinBERT scores — skip", crypto.upper())
        else:
            logger.info("[%s] Queuing for FinBERT inference", crypto.upper())
            to_process.append(crypto)

    if not to_process:
        logger.info("Nothing to process — all cryptos already have FinBERT scores.")
    else:
        logger.info("Will process: %s", to_process)
        # Rough ETA
        n_total = 0
        for crypto in to_process:
            ckpt = DATA_PROCESSED / f"finbert_checkpoint_{crypto}.csv"
            full = DATA_PROCESSED / f"tweets_with_sentiment_{crypto}.csv"
            if ckpt.exists():
                done = len(pd.read_csv(ckpt))
            else:
                done = 0
            total = len(pd.read_csv(full, usecols=["vader_compound"]))
            n_total += (total - done)
        eta_hours = n_total / 12.42 / 3600
        logger.info("Remaining rows: ~%d  ETA: ~%.1f hours (at 12.42 msg/sec)", n_total, eta_hours)

    timings: dict[str, float | None] = {}
    summaries: list[dict] = []

    # ── Phase 1: inference ─────────────────────────────────────────────────
    for crypto in to_process:
        sent_path = DATA_PROCESSED / f"tweets_with_sentiment_{crypto}.csv"
        logger.info("=" * 60)
        logger.info("[%s] Loading %s...", crypto.upper(), sent_path.name)
        df = pd.read_csv(sent_path, low_memory=False)
        logger.info("[%s] Loaded %d rows", crypto.upper(), len(df))

        t0 = time.time()
        finbert_result = finbert_infer_with_checkpoint(
            df["cleaned_text"].fillna("").reset_index(drop=True),
            crypto,
        )
        elapsed = time.time() - t0
        timings[crypto] = elapsed
        logger.info(
            "[%s] Inference done: %.0f sec (%.2f h)", crypto.upper(), elapsed, elapsed / 3600
        )

        # Attach and save
        df = df.copy()
        df["finbert_label"] = finbert_result["finbert_label"].values
        df["finbert_score"] = finbert_result["finbert_score"].values
        df.to_csv(sent_path, index=False)
        logger.info("[%s] Saved updated %s", crypto.upper(), sent_path.name)

        # Clean up checkpoint file now that we're done
        ckpt_path = DATA_PROCESSED / f"finbert_checkpoint_{crypto}.csv"
        if ckpt_path.exists():
            ckpt_path.unlink()
            logger.info("[%s] Checkpoint file removed", crypto.upper())

    # ── Phase 2: comparison outputs ────────────────────────────────────────
    all_cryptos_canonical = ["btc", "eth", "bnb"]
    for crypto in all_cryptos_canonical:
        sent_path = DATA_PROCESSED / f"tweets_with_sentiment_{crypto}.csv"
        if not sent_path.exists():
            continue
        df = pd.read_csv(sent_path, low_memory=False)
        if "finbert_label" not in df.columns:
            logger.warning("[%s] Still no finbert_label — skipping comparison output", crypto.upper())
            continue
        summary = generate_comparison_outputs(df, crypto)
        summaries.append(summary)

    # ── Phase 3: cross-crypto tables & plot ───────────────────────────────
    if summaries:
        generate_cross_crypto_summary(summaries)
        update_comparison_csv(summaries)
    else:
        logger.warning("No summaries available — cross-crypto outputs skipped")

    # ── Update timing log ──────────────────────────────────────────────────
    update_timing_log(timings)

    # ── Phase 4: run pytest ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running pytest to verify all tests pass...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=str(BASE), timeout=300,
        )
        logger.info("pytest stdout:\n%s", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode == 0:
            logger.info("pytest: ALL TESTS PASSED")
        else:
            logger.warning("pytest: SOME TESTS FAILED (returncode=%d)", result.returncode)
            if result.stderr:
                logger.warning("pytest stderr:\n%s", result.stderr[-1000:])
    except subprocess.TimeoutExpired:
        logger.warning("pytest timed out after 300 sec")
    except Exception as exc:
        logger.warning("pytest failed to run: %s", exc)

    # ── Final report ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("DONE. Summary:")
    for s in summaries:
        logger.info(
            "  %-6s  N=%7d  agree=%5.1f%%"
            "  VADER  pos=%5.1f%% neu=%5.1f%% neg=%5.1f%%"
            "  FinBERT pos=%5.1f%% neu=%5.1f%% neg=%5.1f%%",
            s["crypto"], s["n"], s["agreement_pct"],
            s["vader_pos_pct"], s["vader_neu_pct"], s["vader_neg_pct"],
            s["finbert_pos_pct"], s["finbert_neu_pct"], s["finbert_neg_pct"],
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
