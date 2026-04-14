"""Sentiment analysis using VADER and FinBERT."""
from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_finbert_pipeline = None


def _get_finbert():
    """Lazy-load FinBERT pipeline (avoids heavy import at module level)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        logger.info("Loading FinBERT model (ProsusAI/finbert)...")
        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
        logger.info("FinBERT loaded.")
    return _finbert_pipeline


def score_vader(texts: pd.Series) -> pd.DataFrame:
    """Compute VADER compound score for each tweet.

    Uses the *original* (uncleaned) text because VADER leverages
    capitalisation, punctuation, and emoji as tonal signals.

    Args:
        texts: Series of raw tweet strings.

    Returns:
        DataFrame with columns: vader_compound, vader_pos,
        vader_neg, vader_neu.
    """
    analyzer = SentimentIntensityAnalyzer()
    records = []
    for text in texts:
        scores = analyzer.polarity_scores(str(text))
        records.append({
            "vader_compound": scores["compound"],
            "vader_pos": scores["pos"],
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
        })
    return pd.DataFrame(records, index=texts.index)


def score_finbert(texts: pd.Series, batch_size: int = 16) -> pd.DataFrame:
    """Compute FinBERT sentiment label and confidence for each tweet.

    Uses *cleaned* text (URLs and mentions removed). Runs in batches
    for efficiency; falls back gracefully to CPU if no GPU available.

    Args:
        texts: Series of pre-cleaned tweet strings.
        batch_size: Number of texts per inference batch.

    Returns:
        DataFrame with columns: finbert_label (positive/negative/neutral),
        finbert_score (confidence in [0, 1]).
    """
    pipe = _get_finbert()
    text_list = texts.tolist()
    results = []
    for i in tqdm(range(0, len(text_list), batch_size),
                  desc="FinBERT inference"):
        batch = text_list[i: i + batch_size]
        batch = [t if t.strip() else "neutral" for t in batch]
        out = pipe(batch)
        results.extend(out)

    labels = [r["label"].lower() for r in results]
    scores = [r["score"] for r in results]
    return pd.DataFrame(
        {"finbert_label": labels, "finbert_score": scores},
        index=texts.index,
    )


def get_torchinfo_summary() -> str:
    """Return torchinfo summary of the FinBERT model architecture.

    Requirement: FOS 6P — torchinfo output with layer count,
    parameter count, and tensor dimensions.

    Returns:
        String representation of the model summary.
    """
    from torchinfo import summary
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "ProsusAI/finbert"
    logger.info("Loading FinBERT for torchinfo summary...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    dummy = tokenizer(
        "Bitcoin price rises as market sentiment improves",
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )

    col_names = ("input_size", "output_size", "num_params", "trainable")
    model_summary = summary(
        model,
        input_data={
            "input_ids": dummy["input_ids"],
            "attention_mask": dummy["attention_mask"],
        },
        col_names=col_names,
        verbose=0,
    )
    return str(model_summary)


def export_onnx(output_path: str = "results/finbert_architecture.onnx") -> None:
    """Export FinBERT to ONNX format for Netron visualisation.

    Requirement: FOS 6P — architecture graph via Netron.
    The exported file can be opened at https://netron.app.

    Args:
        output_path: Destination path for the ONNX file.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from pathlib import Path

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    dummy = tokenizer(
        "Bitcoin rises",
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding="max_length",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    logger.info("ONNX model saved to %s", output_path)
