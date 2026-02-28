from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np

from conase_geo.models.text_mpsa import TextMPSADenseNetClassifier
from conase_geo.text_utils import encode_text_to_token_ids, normalize_text, scores_to_probs

try:
    import torch
except Exception:  # pragma: no cover - optional runtime path
    torch = None

MODEL_TYPE_MPSA_TOKEN = "text_mpsa_token_map"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict location from writing sample.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to text_model.joblib")
    parser.add_argument("--text", type=str, default="", help="Inline text sample.")
    parser.add_argument("--text_file", type=Path, default=None, help="Optional text file path.")
    parser.add_argument("--top_k", type=int, default=5)
    return parser


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        return args.text_file.read_text(encoding="utf-8")
    return args.text


def _predict_mpsa_token_map(
    *,
    artifact: Dict[str, object],
    clean_text: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    if torch is None:
        raise RuntimeError("PyTorch is required to run MPSA token-map prediction.")
    idx_to_label = artifact["idx_to_label"]
    state_dict = artifact.get("model_state_dict")
    vocab = artifact.get("vocab")
    config = artifact.get("config", {})
    if not isinstance(idx_to_label, list):
        raise ValueError("Checkpoint idx_to_label is invalid.")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint model_state_dict is missing for token-map MPSA model.")
    if not isinstance(vocab, dict):
        raise ValueError("Checkpoint vocab is missing for token-map MPSA model.")
    if not isinstance(config, dict):
        config = {}

    block_cfg_raw = config.get("mpsa_block_config", [6, 12, 24, 16])
    if not isinstance(block_cfg_raw, Sequence) or len(block_cfg_raw) != 4:
        block_cfg_raw = [6, 12, 24, 16]
    block_cfg = tuple(int(v) for v in block_cfg_raw)
    embed_dim = int(config.get("embed_dim", 128))
    max_len = int(config.get("chunk_len", config.get("max_len", 384)))

    model = TextMPSADenseNetClassifier(
        num_classes=len(idx_to_label),
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        max_len=max_len,
        block_config=block_cfg,  # type: ignore[arg-type]
        growth_rate=int(config.get("mpsa_growth_rate", 32)),
        num_init_features=int(config.get("mpsa_num_init_features", 64)),
        bn_size=int(config.get("mpsa_bn_size", 4)),
        drop_rate=float(config.get("mpsa_drop_rate", 0.0)),
        embed_dropout=float(config.get("mpsa_embed_dropout", 0.0)),
        head_hidden_dim=int(config.get("head_hidden_dim", 512)),
        head_dropout=float(config.get("head_dropout", 0.2)),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ids = encode_text_to_token_ids(clean_text, vocab=vocab, max_len=max_len)
    x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        logits = out["accent"] if isinstance(out, dict) else out
        probs_1d = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    k = min(max(1, top_k), len(idx_to_label))
    idx = np.argpartition(probs_1d, -k)[-k:]
    idx = idx[np.argsort(probs_1d[idx])[::-1]]
    return [(str(idx_to_label[i]), float(probs_1d[i])) for i in idx]


def predict_text(
    checkpoint: Path,
    text: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    artifact = joblib.load(checkpoint)
    model_type = str(artifact.get("model_type", ""))
    idx_to_label = artifact["idx_to_label"]

    clean = normalize_text(text)
    if not clean:
        raise ValueError("Input text is empty after normalization.")

    if model_type == MODEL_TYPE_MPSA_TOKEN:
        return _predict_mpsa_token_map(artifact=artifact, clean_text=clean, top_k=top_k)

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    X = vectorizer.transform([clean])
    scores = np.asarray(model.decision_function(X))
    if len(model.classes_) > 2 and scores.ndim == 1:
        scores = scores.reshape(1, -1)
    probs = scores_to_probs(scores)
    probs_1d = probs[0]

    k = min(max(1, top_k), len(idx_to_label))
    idx = np.argpartition(probs_1d, -k)[-k:]
    idx = idx[np.argsort(probs_1d[idx])[::-1]]
    return [(str(idx_to_label[i]), float(probs_1d[i])) for i in idx]


def main() -> None:
    args = _build_arg_parser().parse_args()
    text = _load_text(args)
    preds = predict_text(checkpoint=args.checkpoint, text=text, top_k=args.top_k)
    for rank, (label, prob) in enumerate(preds, start=1):
        print(f"{rank}. {label}\t{prob:.4f}")


if __name__ == "__main__":
    main()
