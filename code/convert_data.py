"""Merge raw tweet CSVs into clean per-crypto datasets.

Reads {crypto}_twitter_extraction*.csv from data/raw/Tweets_extraction_btc/,
applies rigorous quality filters using vectorized pandas operations,
and writes data/raw/tweets_{crypto}.csv. With --crypto all also writes
tweets_all.csv with a 'crypto' column.

Usage:
    python convert_data.py                 # BTC only → tweets_btc.csv
    python convert_data.py --crypto eth    # ETH only → tweets_eth.csv
    python convert_data.py --crypto all    # all three + tweets_all.csv
"""
from __future__ import annotations

import argparse
import glob
import html
import re
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw/Tweets_extraction_btc"
READ_COLS = ["created_at", "id", "full_text", "retweet_count",
             "favorite_count", "lang"]

CRYPTO_CHOICES = ["btc", "eth", "bnb", "all"]


# ─── Vectorized junk detection (fast) ────────────────────────────────────────

# Non-capturing groups (?:...) throughout to avoid pandas warnings
SPAM_PATTERN = "|".join([
    r"earn\s+(?:passive\s+)?income",
    r"(?:earn|win|get|claim)\s+(?:up\s+to\s+)?\$?\d+\s*(?:daily|usd|usdt|hourly)",
    r"reward\s+up\s*to\s+\$?\d+",
    r"(?:daily|hourly)\s+(?:withdraw|profit|return|roi|reward)",
    r"minimum\s+deposit",
    r"staking\s+pool",
    r"multiple\s+staking",
    r"cloud\s+mining",
    r"guaranteed\s+(?:return|profit|income)",
    r"dm\s+me\s+(?:now|for|to|if)",
    r"inbox\s+me",
    r"recover\s+(?:your\s+)?(?:trust\s+)?wallet",
    r"account\s+recovery",
    r"send\s+\d+(?:\.\d+)?\s*(?:btc|eth|bnb)\s+(?:to\s+)?(?:get|receive)",
    r"free\s+(?:airdrop|giveaway|nft|token|crypto)",
    r"(?:airdrop|giveaway)\s+is\s+live",
    r"claim\s+(?:your\s+)?(?:free|airdrop|reward|token)",
    r"join\s+(?:us|our|the)\s+(?:at|on|in)\b.*crypto",
    r"check\s+it\s+out\s*:\s*http",
    r"sign\s+up\s+(?:now|today|here)\s*(?:and|to)\s*(?:get|earn|receive)",
])

BOT_PATTERN = "|".join([
    # Whale / large transaction alerts
    r"whale\s*alert",
    r"bullwhale",
    r"bearwhale",
    r"\d[\d,.]+\s*#?\$?(?:btc|eth|bnb|usdt)\s+\([\d,.]+\s*(?:usd|eur)\)",
    # Price feed / ticker bots
    r"current\s+#?\$?\w+\s+price",
    r"price\s+update",
    r"price\s+alert\s*:",
    r"currency\s+price\s+update",
    r"bitcoin\s+last\s+price",
    r"daily\s+indicators\s*:",
    r"pivot\s+fibonacci",
    r"variation\s+since\s+\d+h\d+",
    r"follow\s+for\s+recent\s+\w+\s+price",
    # Trading signal / position bots
    r"(?:shorted|longed)\s.{0,50}on\s+(?:binance|ftx|bybit|bitmex|okex|huobi|kraken)",
    r"(?:shorted|longed)\s*\(?(?:sell|buy)\)?\s*\[",
    r"just\s+(?:shorted|longed)\s+\$[\d,]+\s+worth",
    r"open\s+@\s*[\d.]+\s+close\s+@\s*[\d.]+\s+with\s+id\s+\d+",
    r"scalping\s+signal",
    r"enter\s+(?:long|short)\s+position\s+for",
    r"\d[\d,.]+\s+(?:btcusdt|ethusdt|bnbusdt|btcusd|btc-perp)\s+(?:shorted|longed)",
    r"\$[\d,.]+\s+(?:btcusdt|ethusdt|bnbusdt|btcusd|btc-perp)\s+(?:shorted|longed)",
    # Volume alert bots
    r"\d+x\s+volume",
    # Liquidation bots
    r"liquidated\s+(?:short|long)",
    r"real\s+time\s+liquidation",
    # Contract / token alert bots
    r"token\s+contract\s*:",
    r"0x[0-9a-f]{10,}",
    # Data dump bots (long number sequences)
    r"(?:\d[\d,.]+\s+){4,}",
    # NFT / OpenSea bots
    r"check\s+out\s+this\s+item\s+on\s+opensea",
    r"(?:sold|bought|listed)\s+for\s+\d+(?:\.\d+)?\s*#?(?:eth|bnb|btc|sol)",
    r"rarity\s+rank\s+\d+",
    # Position / order tracking bots
    r"opened\s+(?:new\s+)?(?:long|short)\s+position",
    r"closed\s+(?:long|short)\s+position",
    r"unusual\s+(?:limit\s+)?order\s.*detected",
    r"prediction\s+value\s+at\s+time",
    # Self-identified price ticker bots
    r"i\s+tweet\s+updated\s+prices",
    r"current\s+data\s+for\s+\w+\s+price",
    # Signal / ticker bots
    r"\w+usdt\s+\w+\s+signal\s+\d+",
    r"change\s+in\s+\d+h\s*:\s*[+-]",
    r"bestask\s+exchange|bestbid\s+exchange",
    r"just\s+transfer(?:ed|red)\s+\d+",
    r"upflow\s+volume\s+now",
    r"market\s*cap\s*:\s*\$[\d,]+",
    # Faucet / engagement bots
    r"faucet\s+.*(?:making|lets?)\s+me\s+tweet",
    r"follow\s+(?:me\s+)?(?:for|and)\s+(?:recent|daily|more)\s+(?:bitcoin|btc|crypto|price|update)",
    r"follow\s+for\s+follow",
])


def flag_junk(series: pd.Series) -> pd.Series:
    """Vectorized junk detection using str.contains (much faster than apply)."""
    s = series.fillna("")
    is_spam = s.str.contains(SPAM_PATTERN, case=False, regex=True, na=False)
    is_bot = s.str.contains(BOT_PATTERN, case=False, regex=True, na=False)
    return is_spam | is_bot


# ─── Text cleaning (vectorized) ──────────────────────────────────────────────

def clean_series(series: pd.Series) -> pd.Series:
    """Vectorized text cleaning for the full_text column.

    Strips URLs, @mentions, hashtag/cashtag symbols, emoji/non-ASCII,
    HTML entities. Lowercases (ProsusAI/finbert is uncased).
    """
    s = series.fillna("")
    s = s.apply(lambda t: html.unescape(t) if t else "")  # HTML decode
    s = s.str.replace(r"http\S+|www\.\S+", "", regex=True)          # URLs
    s = s.str.replace(r"^RT\s+@\w+:?\s*", "", regex=True)           # RT prefix
    s = s.str.replace(r"@\w+", "", regex=True)                      # @mentions
    s = s.str.replace(r"#(\w+)", r"\1", regex=True)                  # #BTC → BTC
    s = s.str.replace(r"\$([A-Za-z]{2,})", r"\1", regex=True)       # $BTC → BTC
    s = s.str.replace(r"[^\x00-\x7F]+", " ", regex=True)            # emoji/non-ASCII
    s = s.str.replace(r'["""]', '"', regex=True)                     # fancy quotes
    s = s.str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return s


def word_count(series: pd.Series) -> pd.Series:
    """Count words (length > 1) in a cleaned text series."""
    return series.str.split().apply(lambda ws: len([w for w in ws if len(w) > 1]) if isinstance(ws, list) else 0)


def normalise_for_dedup(series: pd.Series) -> pd.Series:
    """Normalise text for near-duplicate detection."""
    s = series.fillna("").str.lower()
    s = s.str.replace(r"http\S+", "", regex=True)
    s = s.str.replace(r"@\w+", "", regex=True)
    s = s.str.replace(r"[^a-z0-9 ]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


# ─── Main pipeline ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge raw tweet CSVs into per-crypto datasets")
    p.add_argument("--crypto", choices=CRYPTO_CHOICES, default="btc",
                   help="Cryptocurrency to process (default: btc). 'all' runs all three.")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--out-dir", default="data/raw",
                   help="Output directory for tweets CSV files")
    p.add_argument("--min-words", type=int, default=4,
                   help="Minimum word count after cleaning")
    return p.parse_args()


def process_crypto(prefix: str, raw_dir: str, min_words: int) -> pd.DataFrame:
    """Run the full filter pipeline for a single cryptocurrency prefix.

    Args:
        prefix: Filename prefix, one of 'btc', 'eth', 'bnb'.
        raw_dir: Directory containing raw CSV files.
        min_words: Minimum word count after cleaning.

    Returns:
        Filtered DataFrame ready to be saved.
    """
    # ── 1. Discover files ──────────────────────────────────────────────────
    files = sorted(glob.glob(f"{raw_dir}/{prefix}_twitter_extraction*.csv"))
    logger.info("[%s] Found %d CSV files", prefix.upper(), len(files))
    if not files:
        logger.warning("[%s] No files found in %s — skipping", prefix.upper(), raw_dir)
        return pd.DataFrame()

    # ── 2. Read & concat ───────────────────────────────────────────────────
    chunks, skipped = [], 0
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in READ_COLS,
                             dtype={"id": str}, low_memory=False)
            chunks.append(df[[c for c in READ_COLS if c in df.columns]])
        except Exception as e:
            logger.warning("Skip %s: %s", f, e)
            skipped += 1

    df = pd.concat(chunks, ignore_index=True)
    n0 = len(df)
    logger.info("[%s] Raw rows: %d  (skipped %d files)", prefix.upper(), n0, skipped)

    # ── 3. English only ────────────────────────────────────────────────────
    if "lang" in df.columns:
        before = len(df)
        df = df[df["lang"] == "en"].copy()
        logger.info("[%s] English filter:           %d  (-%d)", prefix.upper(), len(df), before - len(df))

    # ── 4. Drop nulls / empties ────────────────────────────────────────────
    df = df.dropna(subset=["full_text"])
    df = df[df["full_text"].str.strip().astype(bool)].copy()

    # ── 5. Dedup by ID ─────────────────────────────────────────────────────
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="id")
        logger.info("[%s] Dedup by ID:              %d  (-%d)", prefix.upper(), len(df), before - len(df))

    # ── 6. Dedup by normalised text ────────────────────────────────────────
    before = len(df)
    df["_norm"] = normalise_for_dedup(df["full_text"])
    df = df.drop_duplicates(subset="_norm", keep="first")
    df = df.drop(columns=["_norm"])
    logger.info("[%s] Dedup by text:            %d  (-%d)", prefix.upper(), len(df), before - len(df))

    # ── 7. Remove retweets ─────────────────────────────────────────────────
    before = len(df)
    df = df[~df["full_text"].str.strip().str.startswith("RT @")].copy()
    logger.info("[%s] Remove retweets:          %d  (-%d)", prefix.upper(), len(df), before - len(df))

    # ── 8. Remove spam / bots / noise (VECTORIZED) ─────────────────────────
    before = len(df)
    junk_mask = flag_junk(df["full_text"])
    df = df[~junk_mask].copy()
    logger.info("[%s] Remove spam/bots/noise:   %d  (-%d)", prefix.upper(), len(df), before - len(df))

    # ── 9. Parse dates ─────────────────────────────────────────────────────
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True, errors="coerce",
    )
    df = df.dropna(subset=["created_at"])
    df["date"] = df["created_at"].dt.date

    # ── 10. Build text columns ─────────────────────────────────────────────
    df["original_text"] = df["full_text"].apply(
        lambda t: html.unescape(t) if isinstance(t, str) else t
    )
    df["full_text"] = clean_series(df["original_text"])

    # ── 11. Remove short tweets (after cleaning) ───────────────────────────
    before = len(df)
    wc = word_count(df["full_text"])
    df = df[wc >= min_words].copy()
    logger.info("[%s] Min %d words:              %d  (-%d)", prefix.upper(), min_words, len(df), before - len(df))

    df = df[df["full_text"].str.strip().astype(bool)].copy()
    df = df.sort_values("created_at").reset_index(drop=True)

    logger.info("[%s] =" * 1 + "=" * 55, prefix.upper())
    logger.info("[%s] FILTERED: %d tweets | dates %s → %s | %d unique days",
                prefix.upper(), len(df), df["date"].min(), df["date"].max(),
                df["date"].nunique())
    logger.info("[%s] Removed total: %d (%.1f%%)", prefix.upper(),
                n0 - len(df), (1 - len(df)/n0)*100 if n0 else 0)
    logger.info("[%s] =" * 1 + "=" * 55, prefix.upper())

    return df


def main():
    args = parse_args()

    cryptos = ["btc", "eth", "bnb"] if args.crypto == "all" else [args.crypto]
    results: dict[str, pd.DataFrame] = {}

    for crypto in cryptos:
        df = process_crypto(crypto, args.raw_dir, args.min_words)
        if df.empty:
            continue
        out_path = f"{args.out_dir}/tweets_{crypto}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %d tweets to %s", len(df), out_path)
        results[crypto] = df

    if args.crypto == "all" and results:
        parts = []
        for crypto, df in results.items():
            df = df.copy()
            df["crypto"] = crypto
            parts.append(df)
        combined = pd.concat(parts, ignore_index=True).sort_values("created_at")
        combined_path = f"{args.out_dir}/tweets_all.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("Combined all: %d tweets → %s", len(combined), combined_path)

        logger.info("=" * 60)
        logger.info("SUMMARY")
        for crypto, df in results.items():
            logger.info("  %s: %d tweets, %d days", crypto.upper(), len(df), df["date"].nunique())
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
