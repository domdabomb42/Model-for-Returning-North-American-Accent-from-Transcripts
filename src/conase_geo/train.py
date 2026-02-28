from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from conase_geo.config import DEFAULT_PAUSE_THRESHOLD_SECONDS, DEFAULT_SAMPLE_RATE, PROSODY_FEATURE_NAMES, TIMING_FEATURE_NAMES
from conase_geo.data.parse_tokens import parse_token_times_json
from conase_geo.features.prosody_features import compute_prosody_from_file, prosody_feature_vector
from conase_geo.features.timing_features import compute_timing_features, timing_feature_vector
from conase_geo.models.classifier import AudioOnlyClassifier, AudioPlusTimingClassifier, TimingOnlyClassifier
from conase_geo.models.encoders import build_audio_encoder
from conase_geo.models.mpsa_densenet import MPSAConfig, MPSADenseNet

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None

LOGGER = logging.getLogger(__name__)
MODEL_CHOICES = ("timing_only", "audio_only", "audio_plus_timing", "mpsa_densenet")
ENCODER_CHOICES = ("wav2vec2_base", "mfcc")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_block_config(raw: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--mpsa_block_config must have 4 comma-separated integers, e.g. 6,12,24,16")
    values = tuple(int(p) for p in parts)
    if any(v <= 0 for v in values):
        raise ValueError("--mpsa_block_config values must be positive.")
    return values  # type: ignore[return-value]


def load_manifest(manifest_path: Path, model_type: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        raise ValueError("Manifest must contain a 'label' column.")
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"] != ""].copy()

    for col in ["clip_start", "clip_end"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df["clip_start"] = df["clip_start"].fillna(0.0)
    df["clip_end"] = df["clip_end"].fillna(df["clip_start"] + 1.0)

    if "token_times_json" not in df.columns:
        df["token_times_json"] = "[]"
    df["token_times_json"] = df["token_times_json"].fillna("[]")

    if model_type != "timing_only":
        if "audio_path" not in df.columns:
            raise ValueError("Audio model selected but 'audio_path' column is missing.")
        df["audio_path"] = df["audio_path"].astype(str).str.strip()
        exists = df["audio_path"].map(lambda p: Path(p).exists() if p else False)
        missing_count = int((~exists).sum())
        if missing_count > 0:
            LOGGER.warning("Dropping %s rows with missing audio files.", missing_count)
        df = df[exists].copy()

    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError("No usable rows found in manifest after filtering.")
    return df


def extract_timing_features_for_df(df: pd.DataFrame, pause_threshold: float) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Timing features", leave=False):
        token_times = parse_token_times_json(getattr(row, "token_times_json", "[]"))
        clip_start = float(getattr(row, "clip_start", 0.0) or 0.0)
        clip_end = float(getattr(row, "clip_end", clip_start + 1.0) or (clip_start + 1.0))
        clip_duration = max(clip_end - clip_start, 1e-6)

        if token_times and max(token_times) <= clip_duration + 1.0:
            features = compute_timing_features(token_times, clip_start=0.0, clip_end=clip_duration, pause_threshold=pause_threshold)
        else:
            features = compute_timing_features(
                token_times,
                clip_start=clip_start,
                clip_end=clip_end,
                pause_threshold=pause_threshold,
            )
        vectors.append(timing_feature_vector(features, TIMING_FEATURE_NAMES))
    return np.vstack(vectors).astype(np.float32)


def extract_prosody_features_for_df(df: pd.DataFrame, max_audio_seconds: float) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for path_str in tqdm(df["audio_path"].astype(str).tolist(), desc="Prosody features", leave=False):
        path = Path(path_str)
        if path.exists():
            stats = compute_prosody_from_file(path, sample_rate=DEFAULT_SAMPLE_RATE, max_seconds=max_audio_seconds)
        else:
            stats = {}
        vectors.append(prosody_feature_vector(stats, PROSODY_FEATURE_NAMES))
    return np.vstack(vectors).astype(np.float32)


def make_group_splits(df: pd.DataFrame, group_col: str, seed: int) -> Dict[str, np.ndarray]:
    indices = np.arange(len(df))
    if group_col not in df.columns:
        fallback = "channel_id" if "channel_id" in df.columns else "video_id"
        LOGGER.warning("split_by=%s not found. Falling back to %s.", group_col, fallback)
        group_col = fallback

    groups = df[group_col].fillna("").astype(str)
    if groups.nunique() < 3:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(0.8 * n))
        n_val = max(1, int(0.1 * n))
        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train : n_train + n_val]
        test_idx = shuffled[n_train + n_val :]
        if len(test_idx) == 0:
            test_idx = val_idx[-1:]
            val_idx = val_idx[:-1] if len(val_idx) > 1 else val_idx
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(indices, groups=groups))

    train_val_groups = groups.iloc[train_val_idx]
    if train_val_groups.nunique() < 2:
        train_idx = train_val_idx
        val_idx = np.asarray([], dtype=np.int64)
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1)
    train_rel, val_rel = next(val_splitter.split(train_val_idx, groups=train_val_groups))
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_label_mapping(train_labels: Sequence[str]) -> Dict[str, int]:
    labels = sorted(set(train_labels))
    return {label: idx for idx, label in enumerate(labels)}


class ManifestDataset(Dataset):
    def __init__(
        self,
        targets: np.ndarray,
        feature_matrix: Optional[np.ndarray] = None,
        audio_paths: Optional[List[str]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_audio_seconds: float = 10.0,
        audio_frontend: str = "waveform",
        mpsa_n_mfcc: int = 64,
        mpsa_n_fft: int = 1024,
        mpsa_hop_length: int = 256,
        age_targets: Optional[np.ndarray] = None,
        gender_targets: Optional[np.ndarray] = None,
    ) -> None:
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.features = torch.as_tensor(feature_matrix, dtype=torch.float32) if feature_matrix is not None else None
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_audio_seconds)
        self.audio_frontend = audio_frontend
        self.age_targets = torch.as_tensor(age_targets, dtype=torch.long) if age_targets is not None else None
        self.gender_targets = torch.as_tensor(gender_targets, dtype=torch.long) if gender_targets is not None else None
        self.mfcc_transform = None

        if self.audio_paths is not None and torchaudio is None:
            raise RuntimeError("torchaudio is required for audio models.")
        if self.audio_frontend == "mpsa" and torchaudio is not None:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=mpsa_n_mfcc,
                melkwargs={
                    "n_fft": mpsa_n_fft,
                    "hop_length": mpsa_hop_length,
                    "n_mels": mpsa_n_mfcc,
                    "center": True,
                },
            )
        if self.audio_frontend not in {"waveform", "mpsa"}:
            raise ValueError(f"Unsupported audio_frontend: {self.audio_frontend}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.targets.shape[0])

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        assert torchaudio is not None
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        wav = wav.squeeze(0).float()
        if wav.numel() > self.max_samples:
            wav = wav[: self.max_samples]
        elif wav.numel() < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.numel()))
        return wav

    def _waveform_to_mpsa_input(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.mfcc_transform is None:
            raise RuntimeError("MFCC transform is not initialized for mpsa frontend.")
        mfcc = self.mfcc_transform(waveform.unsqueeze(0)).squeeze(0)  # [n_mfcc, frames]
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
        return mfcc.unsqueeze(0).repeat(2, 1, 1).float()  # [2, n_mfcc, frames]

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        wav = self._load_waveform(audio_path)
        if self.audio_frontend == "mpsa":
            return self._waveform_to_mpsa_input(wav)
        return wav

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {"label": self.targets[idx]}
        if self.features is not None:
            item["features"] = self.features[idx]
        if self.audio_paths is not None:
            item["audio"] = self._load_audio(self.audio_paths[idx])
        if self.age_targets is not None:
            item["age_label"] = self.age_targets[idx]
        if self.gender_targets is not None:
            item["gender_label"] = self.gender_targets[idx]
        return item


def build_model(
    model_type: str,
    encoder_name: str,
    num_classes: int,
    feature_dim: int,
    freeze_encoder: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mpsa_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    if model_type == "timing_only":
        return TimingOnlyClassifier(input_dim=feature_dim, num_classes=num_classes)

    if model_type == "mpsa_densenet":
        cfg = mpsa_config or {}
        config = MPSAConfig(
            block_config=tuple(cfg.get("block_config", (6, 12, 24, 16))),
            growth_rate=int(cfg.get("growth_rate", 32)),
            num_init_features=int(cfg.get("num_init_features", 64)),
            bn_size=int(cfg.get("bn_size", 4)),
            drop_rate=float(cfg.get("drop_rate", 0.0)),
            in_channels=int(cfg.get("in_channels", 2)),
            num_age_classes=int(cfg.get("num_age_classes", 0)),
            num_gender_classes=int(cfg.get("num_gender_classes", 0)),
        )
        return MPSADenseNet(num_classes=num_classes, config=config)

    encoder = build_audio_encoder(encoder_name, freeze=freeze_encoder, sample_rate=sample_rate)
    if model_type == "audio_only":
        return AudioOnlyClassifier(audio_encoder=encoder, num_classes=num_classes)
    if model_type == "audio_plus_timing":
        return AudioPlusTimingClassifier(audio_encoder=encoder, timing_dim=feature_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    model_type: str,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    aux_loss_weight: float = 0.2,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    total_correct = 0
    total_top3 = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in data_loader:
            labels = batch["label"].to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs: Optional[Dict[str, torch.Tensor]] = None
            if model_type == "timing_only":
                logits = model(batch["features"].to(device))
            elif model_type == "audio_only":
                logits = model(batch["audio"].to(device))
            elif model_type == "audio_plus_timing":
                logits = model(batch["audio"].to(device), batch["features"].to(device))
            elif model_type == "mpsa_densenet":
                model_out = model(batch["audio"].to(device))
                if isinstance(model_out, dict):
                    outputs = model_out
                    logits = model_out["accent"]
                else:
                    logits = model_out
            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

            loss = criterion(logits, labels)
            if model_type == "mpsa_densenet" and outputs is not None:
                if "age" in outputs and "age_label" in batch:
                    age_targets = batch["age_label"].to(device)
                    age_mask = age_targets >= 0
                    if age_mask.any():
                        loss = loss + aux_loss_weight * criterion(outputs["age"][age_mask], age_targets[age_mask])
                if "gender" in outputs and "gender_label" in batch:
                    gender_targets = batch["gender_label"].to(device)
                    gender_mask = gender_targets >= 0
                    if gender_mask.any():
                        loss = loss + aux_loss_weight * criterion(outputs["gender"][gender_mask], gender_targets[gender_mask])
            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())

            k = min(3, logits.shape[1])
            topk = torch.topk(logits, k=k, dim=1).indices
            total_top3 += int((topk == labels.unsqueeze(1)).any(dim=1).sum().item())

    if total_items == 0:
        return {"loss": 0.0, "accuracy": 0.0, "top3_accuracy": 0.0}
    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
        "top3_accuracy": total_top3 / total_items,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CoNASE dialect location classifiers.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label_type", type=str, default="state")
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="audio_plus_timing")
    parser.add_argument("--encoder", type=str, choices=ENCODER_CHOICES, default="wav2vec2_base")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split_by", type=str, default="channel_id")
    parser.add_argument("--pause_threshold", type=float, default=DEFAULT_PAUSE_THRESHOLD_SECONDS)
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fine_tune_encoder", action="store_true", help="If set, unfreeze wav2vec2 encoder.")
    parser.add_argument("--mpsa_input_seconds", type=float, default=6.0)
    parser.add_argument("--mpsa_n_mfcc", type=int, default=64)
    parser.add_argument("--mpsa_n_fft", type=int, default=1024)
    parser.add_argument("--mpsa_hop_length", type=int, default=256)
    parser.add_argument("--mpsa_growth_rate", type=int, default=32)
    parser.add_argument("--mpsa_block_config", type=str, default="6,12,24,16")
    parser.add_argument("--mpsa_bn_size", type=int, default=4)
    parser.add_argument("--mpsa_drop_rate", type=float, default=0.0)
    parser.add_argument("--mpsa_num_init_features", type=int, default=64)
    parser.add_argument("--mpsa_use_aux_tasks", action="store_true")
    parser.add_argument("--mpsa_age_col", type=str, default="age_label")
    parser.add_argument("--mpsa_gender_col", type=str, default="gender_label")
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    return parser


def _build_aux_mapping(values: Sequence[str], train_idx: np.ndarray) -> Dict[str, int]:
    train_set = set(int(i) for i in train_idx.tolist())
    unique = sorted({values[i] for i in train_set if values[i]})
    return {v: idx for idx, v in enumerate(unique)}


def _build_aux_targets(values: Sequence[str], mapping: Dict[str, int]) -> np.ndarray:
    targets = np.full(len(values), -1, dtype=np.int64)
    for i, value in enumerate(values):
        if value in mapping:
            targets[i] = mapping[value]
    return targets


def main() -> None:
    configure_logging()
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.manifest, model_type=args.model)
    splits = make_group_splits(df=df, group_col=args.split_by, seed=args.seed)
    labels = df["label"].astype(str).tolist()

    label_to_idx = build_label_mapping(df.iloc[splits["train"]]["label"].astype(str).tolist())
    idx_to_label = [label for label, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    # Filter split indices to labels known from train split.
    for split_name in ["train", "val", "test"]:
        split_idx = [i for i in splits[split_name] if labels[i] in label_to_idx]
        splits[split_name] = np.asarray(split_idx, dtype=np.int64)

    if len(splits["train"]) == 0:
        raise ValueError("No train rows left after label filtering.")
    if len(splits["val"]) == 0:
        LOGGER.warning("Validation split is empty; using part of train split as validation fallback.")
        fallback = splits["train"][: max(1, min(128, len(splits["train"]) // 10))]
        splits["val"] = fallback
    if len(splits["test"]) == 0:
        LOGGER.warning("Test split is empty; using validation split as test fallback.")
        splits["test"] = splits["val"]

    LOGGER.info(
        "Rows after filtering -> train: %s, val: %s, test: %s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    feature_names: List[str] = []
    timing_matrix: Optional[np.ndarray] = None
    if args.model in {"timing_only", "audio_plus_timing"}:
        timing_matrix = extract_timing_features_for_df(df, pause_threshold=args.pause_threshold)

    if args.model == "timing_only":
        assert timing_matrix is not None
        feature_matrix: Optional[np.ndarray] = timing_matrix
        feature_names = list(TIMING_FEATURE_NAMES)
    elif args.model == "audio_plus_timing":
        assert timing_matrix is not None
        prosody_matrix = extract_prosody_features_for_df(df, max_audio_seconds=args.max_audio_seconds)
        feature_matrix = np.concatenate([timing_matrix, prosody_matrix], axis=1)
        feature_names = list(TIMING_FEATURE_NAMES + PROSODY_FEATURE_NAMES)
    else:
        feature_matrix = None
        feature_names = []

    audio_paths = df["audio_path"].astype(str).tolist() if args.model != "timing_only" else None
    audio_frontend = "mpsa" if args.model == "mpsa_densenet" else "waveform"
    effective_audio_seconds = args.mpsa_input_seconds if args.model == "mpsa_densenet" else args.max_audio_seconds

    age_targets_all: Optional[np.ndarray] = None
    gender_targets_all: Optional[np.ndarray] = None
    age_mapping: Dict[str, int] = {}
    gender_mapping: Dict[str, int] = {}
    if args.model == "mpsa_densenet" and args.mpsa_use_aux_tasks:
        if args.mpsa_age_col in df.columns:
            age_values = df[args.mpsa_age_col].fillna("").astype(str).str.strip().tolist()
            age_mapping = _build_aux_mapping(age_values, splits["train"])
            age_targets_all = _build_aux_targets(age_values, age_mapping)
            LOGGER.info("MPSA aux age classes: %s", len(age_mapping))
        else:
            LOGGER.warning("Requested --mpsa_use_aux_tasks but %s is not in manifest.", args.mpsa_age_col)
        if args.mpsa_gender_col in df.columns:
            gender_values = df[args.mpsa_gender_col].fillna("").astype(str).str.strip().tolist()
            gender_mapping = _build_aux_mapping(gender_values, splits["train"])
            gender_targets_all = _build_aux_targets(gender_values, gender_mapping)
            LOGGER.info("MPSA aux gender classes: %s", len(gender_mapping))
        else:
            LOGGER.warning("Requested --mpsa_use_aux_tasks but %s is not in manifest.", args.mpsa_gender_col)

    def _dataset_for_split(split_name: str) -> ManifestDataset:
        idx = splits[split_name]
        y = np.asarray([label_to_idx[labels[i]] for i in idx], dtype=np.int64)
        feats = feature_matrix[idx] if feature_matrix is not None else None
        audio = [audio_paths[i] for i in idx] if audio_paths is not None else None
        age = age_targets_all[idx] if age_targets_all is not None else None
        gender = gender_targets_all[idx] if gender_targets_all is not None else None
        return ManifestDataset(
            targets=y,
            feature_matrix=feats,
            audio_paths=audio,
            sample_rate=DEFAULT_SAMPLE_RATE,
            max_audio_seconds=effective_audio_seconds,
            audio_frontend=audio_frontend,
            mpsa_n_mfcc=args.mpsa_n_mfcc,
            mpsa_n_fft=args.mpsa_n_fft,
            mpsa_hop_length=args.mpsa_hop_length,
            age_targets=age,
            gender_targets=gender,
        )

    train_ds = _dataset_for_split("train")
    val_ds = _dataset_for_split("val")
    test_ds = _dataset_for_split("test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    mpsa_config: Optional[Dict[str, Any]] = None
    if args.model == "mpsa_densenet":
        mpsa_config = {
            "block_config": parse_block_config(args.mpsa_block_config),
            "growth_rate": args.mpsa_growth_rate,
            "num_init_features": args.mpsa_num_init_features,
            "bn_size": args.mpsa_bn_size,
            "drop_rate": args.mpsa_drop_rate,
            "in_channels": 2,
            "num_age_classes": len(age_mapping) if age_mapping else 0,
            "num_gender_classes": len(gender_mapping) if gender_mapping else 0,
        }

    model = build_model(
        model_type=args.model,
        encoder_name=args.encoder,
        num_classes=len(label_to_idx),
        feature_dim=(feature_matrix.shape[1] if feature_matrix is not None else 0),
        freeze_encoder=not args.fine_tune_encoder,
        sample_rate=DEFAULT_SAMPLE_RATE,
        mpsa_config=mpsa_config,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            model_type=args.model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            aux_loss_weight=args.aux_loss_weight,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            model_type=args.model,
            device=device,
            criterion=criterion,
            optimizer=None,
            aux_loss_weight=args.aux_loss_weight,
        )
        LOGGER.info(
            "Epoch %s/%s | train loss %.4f acc %.4f top3 %.4f | val loss %.4f acc %.4f top3 %.4f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["top3_accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["top3_accuracy"],
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "encoder_name": args.encoder,
                "label_type": args.label_type,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "feature_names": feature_names,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "max_audio_seconds": effective_audio_seconds,
                "pause_threshold": args.pause_threshold,
                "best_val_accuracy": best_val_acc,
            }
            if mpsa_config is not None:
                checkpoint["mpsa_config"] = mpsa_config
                checkpoint["mpsa_n_mfcc"] = args.mpsa_n_mfcc
                checkpoint["mpsa_n_fft"] = args.mpsa_n_fft
                checkpoint["mpsa_hop_length"] = args.mpsa_hop_length
                checkpoint["aux_mappings"] = {
                    "age": age_mapping,
                    "gender": gender_mapping,
                }
            torch.save(checkpoint, best_checkpoint_path)
            LOGGER.info("Saved new best checkpoint: %s", best_checkpoint_path)

    best_ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = run_epoch(
        model=model,
        data_loader=test_loader,
        model_type=args.model,
        device=device,
        criterion=criterion,
        optimizer=None,
        aux_loss_weight=args.aux_loss_weight,
    )
    LOGGER.info(
        "Test metrics | loss %.4f acc %.4f top3 %.4f",
        test_metrics["loss"],
        test_metrics["accuracy"],
        test_metrics["top3_accuracy"],
    )

    mapping_path = output_dir / "label_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)

    df_with_split = df.copy()
    df_with_split["split"] = "train"
    df_with_split.loc[splits["val"], "split"] = "val"
    df_with_split.loc[splits["test"], "split"] = "test"
    df_with_split.to_csv(output_dir / "manifest_with_splits.csv", index=False)

    summary = {
        "val_best_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_top3_accuracy": test_metrics["top3_accuracy"],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
