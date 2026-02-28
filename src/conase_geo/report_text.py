from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create PDF evaluation report for writing-only model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to text_model.joblib")
    parser.add_argument("--out_pdf", type=Path, default=None, help="Output PDF path")
    parser.add_argument("--title", type=str, default="Writing-Only Geolocation Evaluation Report")
    parser.add_argument(
        "--history_path",
        type=Path,
        default=None,
        help="Optional epoch history path (.csv, .json, or .jsonl). Defaults to checkpoint directory.",
    )
    return parser


def _metric(metrics: Dict[str, float], key: str) -> float:
    return float(metrics.get(key, 0.0))


def _summary_lines(
    n_classes: int,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    history_df: Optional[pd.DataFrame],
) -> list[str]:
    chance_top1 = 1.0 / max(1, n_classes)
    chance_top3 = min(3, n_classes) / max(1, n_classes)
    uniform_logloss = float(np.log(max(2, n_classes)))

    val_acc = _metric(val_metrics, "accuracy")
    test_acc = _metric(test_metrics, "accuracy")
    val_top3 = _metric(val_metrics, "top3_accuracy")
    test_top3 = _metric(test_metrics, "top3_accuracy")
    val_loss = _metric(val_metrics, "log_loss")
    test_loss = _metric(test_metrics, "log_loss")

    lines = [
        f"Classes: {n_classes}",
        f"Chance baseline top-1: {chance_top1:.2%}",
        f"Chance baseline top-3: {chance_top3:.2%}",
        f"Uniform-probability log-loss baseline: {uniform_logloss:.3f}",
        "",
        f"Validation: acc={val_acc:.2%}, top3={val_top3:.2%}, log-loss={val_loss:.3f}, n={int(_metric(val_metrics, 'n'))}",
        f"Test:       acc={test_acc:.2%}, top3={test_top3:.2%}, log-loss={test_loss:.3f}, n={int(_metric(test_metrics, 'n'))}",
        "",
    ]

    if test_acc <= 0.001:
        lines.append("Observation: test top-1 is effectively 0%; model is failing to generalize to held-out channels.")
    if test_top3 < chance_top3:
        lines.append("Observation: test top-3 is below chance baseline.")
    elif test_top3 < (chance_top3 * 1.25):
        lines.append("Observation: test top-3 is only slightly above chance baseline.")
    else:
        lines.append("Observation: test top-3 is meaningfully above chance baseline.")

    if test_loss > uniform_logloss * 2.0:
        lines.append("Observation: log-loss is very high (confidently wrong predictions).")
    elif test_loss > uniform_logloss * 1.2:
        lines.append("Observation: log-loss is above uniform baseline (weak confidence calibration).")
    else:
        lines.append("Observation: log-loss is near/under uniform baseline.")

    lines.append("")
    lines.append("Suggested next run: more rows, more epochs, and label balancing / stricter group splits.")
    if history_df is not None and not history_df.empty:
        n_epochs = int(history_df["epoch"].nunique()) if "epoch" in history_df.columns else int(len(history_df))
        lines.append(f"Epoch history rows available: {len(history_df)} (epochs: {n_epochs})")
    return lines


def _read_history_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if not rows:
                return None
            return pd.DataFrame(rows)
        if suffix == ".json":
            return pd.read_json(path)
    except Exception:
        return None
    return None


def _load_epoch_history(checkpoint: Path, history_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if history_path is not None:
        df = _read_history_file(history_path)
        if df is not None and not df.empty:
            return df
        return None

    candidates = [
        checkpoint.parent / "epoch_history.csv",
        checkpoint.parent / "epoch_history.jsonl",
        checkpoint.parent / "epoch_history.json",
    ]
    for candidate in candidates:
        df = _read_history_file(candidate)
        if df is not None and not df.empty:
            return df
    return None


def _plot_training_history(pdf: PdfPages, history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return
    df = history_df.copy()
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df[df["epoch"].notna()].copy()
        df["epoch"] = df["epoch"].astype(int)
        df = df.sort_values("epoch")
    else:
        df["epoch"] = np.arange(1, len(df) + 1)

    # Page: train throughput
    if "train_samples_per_sec" in df.columns:
        y = pd.to_numeric(df["train_samples_per_sec"], errors="coerce")
        if y.notna().any():
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.plot(df["epoch"], y, marker="o", linewidth=2, color="#2a9d8f")
            ax.set_title("Training Throughput Over Time")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train Samples / Second")
            ax.grid(alpha=0.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Page: train loss curves
    has_loss = any(col in df.columns for col in ["train_loss_mean", "train_loss_min", "train_loss_max"])
    if has_loss:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        plotted = False
        for col, color, label in [
            ("train_loss_mean", "#1d3557", "Train Loss Mean"),
            ("train_loss_min", "#457b9d", "Train Loss Min"),
            ("train_loss_max", "#e63946", "Train Loss Max"),
        ]:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    ax.plot(df["epoch"], y, marker="o", linewidth=2, label=label, color=color)
                    plotted = True
        if plotted:
            ax.set_title("Training Loss Over Time")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # Page: validation metrics over epochs (if available)
    has_val = any(col in df.columns for col in ["val_accuracy", "val_top3_accuracy", "val_log_loss"])
    if has_val:
        fig, ax1 = plt.subplots(figsize=(11, 8.5))
        plotted_left = False
        for col, color, label in [
            ("val_accuracy", "#264653", "Val Top-1"),
            ("val_top3_accuracy", "#2a9d8f", "Val Top-3"),
        ]:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    ax1.plot(df["epoch"], y, marker="o", linewidth=2, label=label, color=color)
                    plotted_left = True
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.grid(alpha=0.25)

        ax2 = ax1.twinx()
        plotted_right = False
        if "val_log_loss" in df.columns:
            y_loss = pd.to_numeric(df["val_log_loss"], errors="coerce")
            if y_loss.notna().any():
                ax2.plot(df["epoch"], y_loss, marker="s", linewidth=2, color="#e76f51", label="Val Log-Loss")
                ax2.set_ylabel("Log-Loss")
                plotted_right = True

        handles, labels = ax1.get_legend_handles_labels()
        if plotted_right:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            ax1.legend(handles, labels, loc="best")
        ax1.set_title("Validation Metrics Over Time")
        if plotted_left or plotted_right:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_pdf_report(checkpoint: Path, out_pdf: Path, title: str, history_path: Optional[Path] = None) -> Path:
    artifact = joblib.load(checkpoint)
    metrics = artifact.get("metrics", {})
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    idx_to_label = artifact.get("idx_to_label", [])
    config = artifact.get("config", {})
    history_df = _load_epoch_history(checkpoint=checkpoint, history_path=history_path)

    n_classes = int(len(idx_to_label))
    chance_top1 = 1.0 / max(1, n_classes)
    chance_top3 = min(3, n_classes) / max(1, n_classes)
    uniform_logloss = float(np.log(max(2, n_classes)))

    val_acc = _metric(val_metrics, "accuracy")
    test_acc = _metric(test_metrics, "accuracy")
    val_top3 = _metric(val_metrics, "top3_accuracy")
    test_top3 = _metric(test_metrics, "top3_accuracy")
    val_loss = _metric(val_metrics, "log_loss")
    test_loss = _metric(test_metrics, "log_loss")
    val_n = _metric(val_metrics, "n")
    test_n = _metric(test_metrics, "n")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # Page 1: textual summary
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=18, pad=16)

        lines = _summary_lines(n_classes=n_classes, val_metrics=val_metrics, test_metrics=test_metrics, history_df=history_df)
        cfg_text = (
            "Config: "
            f"analyzer={config.get('analyzer')} | "
            f"ngram_range={tuple(config.get('ngram_range', []))} | "
            f"n_features={config.get('n_features')} | "
            f"epochs={config.get('epochs')} | "
            f"label_col={config.get('label_col')} | "
            f"group_col={config.get('group_col')}"
        )
        body = "\n".join(lines + ["", cfg_text])
        ax.text(0.03, 0.96, body, va="top", ha="left", fontsize=12, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: accuracy chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width / 2, [val_acc, test_acc], width=width, label="Top-1 Accuracy")
        ax.bar(x + width / 2, [val_top3, test_top3], width=width, label="Top-3 Accuracy")
        ax.axhline(chance_top1, color="gray", linestyle="--", linewidth=1.5, label="Chance Top-1")
        ax.axhline(chance_top3, color="black", linestyle=":", linewidth=1.5, label="Chance Top-3")
        ax.set_xticks(x, ["Validation", "Test"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Baseline")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: log-loss chart
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.bar(["Validation", "Test"], [val_loss, test_loss], color=["#4C72B0", "#DD8452"])
        ax.axhline(uniform_logloss, color="gray", linestyle="--", linewidth=1.5, label="Uniform Baseline")
        ax.set_ylabel("Log-Loss (lower is better)")
        ax.set_title("Log-Loss Comparison")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: sample counts
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.bar(["Validation", "Test"], [val_n, test_n], color=["#55A868", "#C44E52"])
        ax.set_ylabel("Number of Samples")
        ax.set_title("Evaluation Sample Counts")
        ax.grid(axis="y", alpha=0.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if history_df is not None and not history_df.empty:
            _plot_training_history(pdf, history_df)

    return out_pdf


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_pdf = args.out_pdf or args.checkpoint.with_name("text_evaluation_report.pdf")
    pdf_path = build_pdf_report(
        checkpoint=args.checkpoint,
        out_pdf=out_pdf,
        title=args.title,
        history_path=args.history_path,
    )
    print(f"Saved report: {pdf_path}")


if __name__ == "__main__":
    main()
