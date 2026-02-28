from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a visual PDF report and recommendations from evaluate.py outputs."
    )
    parser.add_argument("--eval_dir", type=Path, required=True, help="Directory containing metrics.json and related files.")
    parser.add_argument("--out_pdf", type=Path, default=None, help="Output PDF path.")
    parser.add_argument(
        "--out_recommendations",
        type=Path,
        default=None,
        help="Output text file path for recommended changes.",
    )
    parser.add_argument("--title", type=str, default="Text Region Classifier Evaluation Report")
    parser.add_argument("--top_confusions_n", type=int, default=20)
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def _class_count(eval_dir: Path) -> Optional[int]:
    cm_path = eval_dir / "confusion_matrix.npy"
    if cm_path.exists():
        try:
            cm = np.load(cm_path)
            if cm.ndim == 2 and cm.shape[0] == cm.shape[1]:
                return int(cm.shape[0])
        except Exception:
            return None
    return None


def _mk_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=14)
    ax.text(0.03, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _reliability_page(pdf: PdfPages, calibration: Dict[str, Any]) -> None:
    bins = calibration.get("bins", [])
    if not isinstance(bins, list) or not bins:
        return
    df = pd.DataFrame(bins)
    if df.empty:
        return
    x = ((df["low"] + df["high"]) / 2.0).to_numpy(dtype=float)
    y_acc = df["accuracy"].to_numpy(dtype=float)
    y_conf = df["avg_conf"].to_numpy(dtype=float)
    counts = df["count"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax1.plot(x, y_acc, marker="o", linewidth=2, label="Accuracy")
    ax1.plot(x, y_conf, marker="s", linewidth=2, label="Confidence")
    ax1.set_title(f"Reliability Diagram (ECE={float(calibration.get('ece', 0.0)):.4f})")
    ax1.set_xlabel("Confidence bin center")
    ax1.set_ylabel("Value")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best")

    ax2.bar(x, counts, width=0.05, color="#5e60ce")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence bin center")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.25)
    ax2.set_title("Prediction count per bin")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _coverage_page(pdf: PdfPages, coverage_csv: Path) -> None:
    if not coverage_csv.exists():
        return
    try:
        df = pd.read_csv(coverage_csv)
    except Exception:
        return
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(11, 8.5))
    ax1.plot(df["threshold"], df["accuracy"], marker="o", linewidth=2, color="#1d3557", label="Accuracy@coverage")
    ax1.plot(df["threshold"], df["macroF1"], marker="s", linewidth=2, color="#2a9d8f", label="MacroF1@coverage")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Score")
    ax1.grid(alpha=0.25)
    ax1.set_title("Coverage vs Accuracy/F1")

    ax2 = ax1.twinx()
    ax2.plot(df["threshold"], df["coverage"], marker="^", linewidth=2, color="#e76f51", label="Coverage")
    ax2.set_ylabel("Coverage")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _confusion_pages(pdf: PdfPages, eval_dir: Path, per_class_df: Optional[pd.DataFrame], top_n: int = 25) -> None:
    cm_path = eval_dir / "confusion_matrix.npy"
    if not cm_path.exists():
        return
    try:
        cm = np.load(cm_path)
    except Exception:
        return
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return
    n = cm.shape[0]
    labels: List[str] = [str(i) for i in range(n)]
    if per_class_df is not None and not per_class_df.empty and "label" in per_class_df.columns:
        labels = per_class_df["label"].astype(str).tolist()
        if len(labels) != n:
            labels = [str(i) for i in range(n)]

    # Full matrix
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title("Confusion Matrix (Counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Top-support subset normalized by row
    support = cm.sum(axis=1)
    order = np.argsort(support)[::-1][: min(top_n, n)]
    cm_sub = cm[np.ix_(order, order)].astype(float)
    row_sum = np.clip(cm_sub.sum(axis=1, keepdims=True), 1.0, None)
    cm_norm = cm_sub / row_sum

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Top-{len(order)} Support Classes (Row-normalized confusion)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    tick_labels = [labels[i][:18] for i in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _per_class_pages(pdf: PdfPages, per_class_csv: Path) -> Optional[pd.DataFrame]:
    if not per_class_csv.exists():
        return None
    try:
        df = pd.read_csv(per_class_csv)
    except Exception:
        return None
    if df.empty:
        return None

    numeric_cols = ["support", "precision", "recall", "f1", "top3_recall", "top5_recall"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0.0)

    # Scatter support vs f1
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.scatter(df["support"], df["f1"], alpha=0.7, s=22, color="#2a9d8f")
    ax.set_xscale("log")
    ax.set_xlabel("Support (log scale)")
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 vs Support")
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Top/bottom classes by F1
    top = df.nlargest(15, "f1").copy()
    bot = df.nsmallest(15, "f1").copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8.5))
    ax1.barh(top["label"].astype(str), top["f1"], color="#457b9d")
    ax1.set_title("Top 15 classes by F1")
    ax1.set_xlabel("F1")
    ax1.invert_yaxis()
    ax1.grid(alpha=0.2, axis="x")

    ax2.barh(bot["label"].astype(str), bot["f1"], color="#e63946")
    ax2.set_title("Bottom 15 classes by F1")
    ax2.set_xlabel("F1")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.2, axis="x")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return df


def _top_confusions_page(pdf: PdfPages, top_confusions_csv: Path, top_n: int) -> None:
    if not top_confusions_csv.exists():
        return
    try:
        df = pd.read_csv(top_confusions_csv)
    except Exception:
        return
    if df.empty:
        return
    if "scope" in df.columns:
        df = df[df["scope"] == "global"].copy()
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df.nlargest(top_n, "count")
    if df.empty:
        return

    pairs = df["true_label"].astype(str) + " -> " + df["pred_label"].astype(str)
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.barh(pairs[::-1], df["count"].to_numpy()[::-1], color="#8d99ae")
    ax.set_title(f"Top {len(df)} Confusion Pairs")
    ax.set_xlabel("Count")
    ax.grid(alpha=0.2, axis="x")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _slices_pages(pdf: PdfPages, slices_json: Path) -> None:
    data = _load_json(slices_json)
    if not data:
        return

    # Length buckets
    length = data.get("length_buckets", {})
    if isinstance(length, dict) and length:
        keys = list(length.keys())
        top1 = [float(length[k].get("top1", 0.0)) for k in keys]
        top5 = [float(length[k].get("top5", 0.0)) for k in keys]
        f1 = [float(length[k].get("macroF1", 0.0)) for k in keys]

        x = np.arange(len(keys))
        w = 0.25
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.bar(x - w, top1, width=w, label="Top1")
        ax.bar(x, top5, width=w, label="Top5")
        ax.bar(x + w, f1, width=w, label="MacroF1")
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        ax.set_ylim(0, 1)
        ax.set_title("Slice Metrics: Transcript Length Buckets")
        ax.grid(alpha=0.2, axis="y")
        ax.legend(loc="best")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    # ASR quantiles
    asr_q = data.get("asr_score_quantiles", {})
    if isinstance(asr_q, dict):
        buckets = asr_q.get("buckets", {})
        if isinstance(buckets, dict) and buckets:
            keys = list(buckets.keys())
            top1 = [float(buckets[k].get("top1", 0.0)) for k in keys]
            top5 = [float(buckets[k].get("top5", 0.0)) for k in keys]
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.plot(keys, top1, marker="o", linewidth=2, label="Top1")
            ax.plot(keys, top5, marker="s", linewidth=2, label="Top5")
            ax.set_ylim(0, 1)
            ax.set_title("Slice Metrics: ASR Artifact Score Quantiles")
            ax.grid(alpha=0.25)
            ax.legend(loc="best")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Confidence buckets
    conf = data.get("confidence_buckets", {})
    if isinstance(conf, dict) and conf:
        keys = list(conf.keys())
        top1 = [float(conf[k].get("top1", 0.0)) for k in keys]
        top5 = [float(conf[k].get("top5", 0.0)) for k in keys]
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.plot(keys, top1, marker="o", linewidth=2, label="Top1")
        ax.plot(keys, top5, marker="s", linewidth=2, label="Top5")
        ax.set_ylim(0, 1)
        ax.set_title("Slice Metrics: Confidence Buckets")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _masked_delta_page(pdf: PdfPages, eval_dir: Path) -> None:
    delta_path = eval_dir / "metrics_delta.json"
    if not delta_path.exists():
        return
    delta = _load_json(delta_path)
    if not delta:
        return
    keys = ["delta_top1", "delta_top5", "delta_macroF1", "delta_ECE", "delta_NLL"]
    vals = [float(delta.get(k, 0.0)) for k in keys]
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    colors = ["#2a9d8f" if v >= 0 else "#e63946" for v in vals]
    ax.bar(keys, vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Entity Mask Stress Test: Metric Deltas (masked - base)")
    ax.set_ylabel("Delta")
    ax.grid(alpha=0.2, axis="y")
    plt.xticks(rotation=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _build_recommendations(
    metrics: Dict[str, Any],
    coverage_df: Optional[pd.DataFrame],
    per_class_df: Optional[pd.DataFrame],
    eval_dir: Path,
) -> List[str]:
    recs: List[str] = []
    top1 = _safe_metric(metrics, "top1_accuracy")
    top5 = _safe_metric(metrics, "top5_accuracy")
    macro_f1 = _safe_metric(metrics, "macro_f1")
    weighted_f1 = _safe_metric(metrics, "weighted_f1")
    ece = _safe_metric(metrics, "ece")
    nll = _safe_metric(metrics, "nll")
    brier = _safe_metric(metrics, "brier")
    samples_per_sec = _safe_metric(metrics, "samples_per_sec")

    n_classes = _class_count(eval_dir)
    if n_classes is None:
        n_classes = 64
    uniform_nll = math.log(max(2, n_classes))

    if top1 < 0.20:
        recs.append(
            "Top-1 is low (<20%). Increase effective context quality: keep max_len=384+, filter very short/low-information transcripts, and consider light text cleanup of ASR artifacts."
        )
    if macro_f1 < 0.10 and weighted_f1 > macro_f1 * 1.8:
        recs.append(
            "Large macro-vs-weighted F1 gap indicates class imbalance. Use class-balanced sampling and/or class-weighted loss to improve tail-region performance."
        )
    if ece > 0.05:
        recs.append(
            "Calibration error is moderate/high. Apply temperature scaling on a validation split before downstream confidence-based decisions."
        )
    if nll > uniform_nll * 1.2 or brier > 0.9:
        recs.append(
            "Probabilities are still noisy (high NLL/Brier). Lower learning rate late in training or use cosine decay with longer tail and checkpoint averaging."
        )
    if samples_per_sec < 30:
        recs.append(
            "Inference throughput is relatively low. For faster eval runs, use --fast, --max_eval_samples for iterations, and optionally reduce batch_size only if OOM occurs."
        )

    if coverage_df is not None and not coverage_df.empty:
        cov_05 = coverage_df.loc[(coverage_df["threshold"] == 0.5)]
        if not cov_05.empty:
            cov = float(cov_05.iloc[0]["coverage"])
            acc = float(cov_05.iloc[0]["accuracy"])
            if acc - top1 > 0.25 and cov < 0.15:
                recs.append(
                    "Confidence thresholding strongly improves precision at low coverage; consider an abstain policy for production predictions."
                )

    if per_class_df is not None and not per_class_df.empty:
        low = per_class_df[per_class_df["f1"] < 0.03]
        if len(low) > 10:
            recs.append(
                "Many classes have near-zero F1. Inspect top_confusions.csv and merge ambiguous labels or add targeted data for repeatedly confused region pairs."
            )
        tiny = per_class_df[per_class_df["support"] < 100]
        if len(tiny) > 0:
            recs.append(
                "Some classes have very low support (<100). Either gather more data for those regions or mark them as low-confidence classes."
            )

    delta = _load_json(eval_dir / "metrics_delta.json")
    if delta:
        d_top1 = float(delta.get("delta_top1", 0.0))
        if d_top1 < -0.03:
            recs.append(
                "Entity masking causes a material Top-1 drop, suggesting location leakage. Strengthen masking or adversarial training against named entities."
            )

    if not recs:
        recs.append("Current metrics are reasonably consistent; continue training longer and monitor macro F1/coverage trade-offs.")
    return recs


def build_visual_report(
    eval_dir: Path,
    out_pdf: Path,
    out_recommendations: Path,
    title: str,
    top_confusions_n: int,
) -> Tuple[Path, Path]:
    metrics_payload = _load_json(eval_dir / "metrics.json")
    metrics = metrics_payload.get("metrics", {}) if isinstance(metrics_payload, dict) else {}
    args = metrics_payload.get("args", {}) if isinstance(metrics_payload, dict) else {}

    calibration = _load_json(eval_dir / "calibration.json")
    slices = _load_json(eval_dir / "slices.json")
    coverage_csv = eval_dir / "coverage_curve.csv"
    per_class_csv = eval_dir / "per_class.csv"
    top_confusions_csv = eval_dir / "top_confusions.csv"

    per_class_df = None
    if per_class_csv.exists():
        try:
            per_class_df = pd.read_csv(per_class_csv)
        except Exception:
            per_class_df = None
    coverage_df = None
    if coverage_csv.exists():
        try:
            coverage_df = pd.read_csv(coverage_csv)
        except Exception:
            coverage_df = None

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_recommendations.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_pdf) as pdf:
        # Page 1 summary text
        summary_lines = [
            f"Checkpoint: {metrics_payload.get('checkpoint_name', '')}",
            f"Split requested: {metrics.get('split_requested', '')}",
            f"Split used: {metrics.get('split_used', '')} ({metrics.get('split_source', '')})",
            f"Samples evaluated: {int(_safe_metric(metrics, 'n_samples'))}",
            f"Wall time (sec): {_safe_metric(metrics, 'wall_time_sec'):.2f}",
            f"Throughput (samples/sec): {_safe_metric(metrics, 'samples_per_sec'):.2f}",
            "",
            f"Top-1: {_safe_metric(metrics, 'top1_accuracy'):.4f}",
            f"Top-3: {_safe_metric(metrics, 'top3_accuracy'):.4f}",
            f"Top-5: {_safe_metric(metrics, 'top5_accuracy'):.4f}",
            f"Macro F1: {_safe_metric(metrics, 'macro_f1'):.4f}",
            f"Weighted F1: {_safe_metric(metrics, 'weighted_f1'):.4f}",
            f"Balanced Accuracy: {_safe_metric(metrics, 'balanced_accuracy'):.4f}",
            f"NLL: {_safe_metric(metrics, 'nll'):.4f}",
            f"Brier: {_safe_metric(metrics, 'brier'):.4f}",
            f"ECE: {_safe_metric(metrics, 'ece'):.4f}",
            "",
            "Run args:",
            f"  batch_size={args.get('batch_size')}  device={args.get('device')}  amp={args.get('amp')}  compile={args.get('compile')}",
            f"  max_eval_samples={args.get('max_eval_samples')}  fast={args.get('fast')}",
        ]
        _mk_text_page(pdf, title, summary_lines)

        # Page 2 global metrics bars
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        names = ["top1", "top3", "top5", "macro_f1", "weighted_f1", "balanced_acc"]
        vals = [
            _safe_metric(metrics, "top1_accuracy"),
            _safe_metric(metrics, "top3_accuracy"),
            _safe_metric(metrics, "top5_accuracy"),
            _safe_metric(metrics, "macro_f1"),
            _safe_metric(metrics, "weighted_f1"),
            _safe_metric(metrics, "balanced_accuracy"),
        ]
        ax.bar(names, vals, color=["#457b9d", "#457b9d", "#457b9d", "#2a9d8f", "#2a9d8f", "#e76f51"])
        ax.set_ylim(0, 1)
        ax.set_title("Global Metrics")
        ax.grid(alpha=0.25, axis="y")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3 uncertainty metrics
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        names = ["NLL", "Brier", "ECE", "OverconfErr"]
        vals = [
            _safe_metric(metrics, "nll"),
            _safe_metric(metrics, "brier"),
            _safe_metric(metrics, "ece"),
            _safe_metric(metrics, "overconfident_error_rate"),
        ]
        ax.bar(names, vals, color="#8d99ae")
        ax.set_title("Uncertainty and Calibration Summary")
        ax.grid(alpha=0.25, axis="y")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _reliability_page(pdf, calibration)
        _coverage_page(pdf, coverage_csv)
        _confusion_pages(pdf, eval_dir, per_class_df, top_n=25)
        per_class_df_final = _per_class_pages(pdf, per_class_csv)
        _top_confusions_page(pdf, top_confusions_csv, top_confusions_n)
        _slices_pages(pdf, eval_dir / "slices.json")
        _masked_delta_page(pdf, eval_dir)

        # Recommendations page
        recs = _build_recommendations(metrics, coverage_df, per_class_df_final, eval_dir)
        rec_lines = [f"{i+1}. {r}" for i, r in enumerate(recs)]
        _mk_text_page(pdf, "Recommended Changes", rec_lines)

    recommendations = _build_recommendations(metrics, coverage_df, per_class_df, eval_dir)
    out_recommendations.write_text("\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations)) + "\n", encoding="utf-8")
    return out_pdf, out_recommendations


def main() -> None:
    args = _build_arg_parser().parse_args()
    eval_dir = args.eval_dir
    out_pdf = args.out_pdf or (eval_dir / "eval_visual_report.pdf")
    out_rec = args.out_recommendations or (eval_dir / "recommended_changes.txt")
    pdf_path, rec_path = build_visual_report(
        eval_dir=eval_dir,
        out_pdf=out_pdf,
        out_recommendations=out_rec,
        title=args.title,
        top_confusions_n=int(args.top_confusions_n),
    )
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved recommendations: {rec_path}")


if __name__ == "__main__":
    main()

