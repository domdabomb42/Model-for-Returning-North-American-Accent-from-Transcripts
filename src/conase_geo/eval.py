from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from conase_geo.config import PROSODY_FEATURE_NAMES, TIMING_FEATURE_NAMES
from conase_geo.train import (
    ManifestDataset,
    build_model,
    extract_prosody_features_for_df,
    extract_timing_features_for_df,
    load_manifest,
    run_epoch,
)

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a manifest.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint["model_type"]
    encoder_name = checkpoint.get("encoder_name", "mfcc")
    label_to_idx = checkpoint["label_to_idx"]
    pause_threshold = float(checkpoint.get("pause_threshold", 0.35))
    sample_rate = int(checkpoint.get("sample_rate", 16_000))
    max_audio_seconds = float(checkpoint.get("max_audio_seconds", 10.0))
    mpsa_config: Dict[str, Any] = dict(checkpoint.get("mpsa_config", {}))
    mpsa_n_mfcc = int(checkpoint.get("mpsa_n_mfcc", 64))
    mpsa_n_fft = int(checkpoint.get("mpsa_n_fft", 1024))
    mpsa_hop_length = int(checkpoint.get("mpsa_hop_length", 256))

    df = load_manifest(args.manifest, model_type=model_type)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("Requested --split but manifest does not contain a split column.")
        df = df[df["split"] == args.split].copy()
    if df.empty:
        raise ValueError("No rows available for evaluation.")

    labels = df["label"].astype(str).tolist()
    keep_idx = [i for i, lab in enumerate(labels) if lab in label_to_idx]
    if not keep_idx:
        raise ValueError("No rows match labels seen by checkpoint.")
    df = df.iloc[keep_idx].reset_index(drop=True)
    labels = df["label"].astype(str).tolist()
    targets = np.asarray([label_to_idx[lab] for lab in labels], dtype=np.int64)

    if model_type == "timing_only":
        feature_matrix = extract_timing_features_for_df(df, pause_threshold=pause_threshold)
        audio_paths = None
        feature_dim = len(TIMING_FEATURE_NAMES)
        audio_frontend = "waveform"
    elif model_type == "audio_plus_timing":
        timing = extract_timing_features_for_df(df, pause_threshold=pause_threshold)
        prosody = extract_prosody_features_for_df(df, max_audio_seconds=max_audio_seconds)
        feature_matrix = np.concatenate([timing, prosody], axis=1)
        audio_paths = df["audio_path"].astype(str).tolist()
        feature_dim = len(TIMING_FEATURE_NAMES) + len(PROSODY_FEATURE_NAMES)
        audio_frontend = "waveform"
    elif model_type == "mpsa_densenet":
        feature_matrix = None
        audio_paths = df["audio_path"].astype(str).tolist()
        feature_dim = 0
        audio_frontend = "mpsa"
    else:
        feature_matrix = None
        audio_paths = df["audio_path"].astype(str).tolist()
        feature_dim = 0
        audio_frontend = "waveform"

    dataset = ManifestDataset(
        targets=targets,
        feature_matrix=feature_matrix,
        audio_paths=audio_paths,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
        audio_frontend=audio_frontend,
        mpsa_n_mfcc=mpsa_n_mfcc,
        mpsa_n_fft=mpsa_n_fft,
        mpsa_hop_length=mpsa_hop_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(
        model_type=model_type,
        encoder_name=encoder_name,
        num_classes=len(label_to_idx),
        feature_dim=feature_dim,
        freeze_encoder=True,
        sample_rate=sample_rate,
        mpsa_config=mpsa_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = run_epoch(
        model=model,
        data_loader=loader,
        model_type=model_type,
        device=device,
        criterion=nn.CrossEntropyLoss(),
        optimizer=None,
    )

    LOGGER.info(
        "Eval metrics | loss %.4f | acc %.4f | top3 %.4f",
        metrics["loss"],
        metrics["accuracy"],
        metrics["top3_accuracy"],
    )


if __name__ == "__main__":
    main()
