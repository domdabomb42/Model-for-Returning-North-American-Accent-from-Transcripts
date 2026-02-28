from __future__ import annotations

import argparse
from pathlib import Path

from conase_geo.predict_text import predict_text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play location guessing from writing.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    return parser


def _read_multiline_text() -> str:
    print("Paste writing sample. End with a blank line:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    args = _build_arg_parser().parse_args()
    text = _read_multiline_text()
    preds = predict_text(checkpoint=args.checkpoint, text=text, top_k=args.top_k)
    if not preds:
        print("No predictions available.")
        return
    best_label, best_prob = preds[0]
    print(f"\nGuess: {best_label} ({best_prob:.2%})")
    print("Top predictions:")
    for rank, (label, prob) in enumerate(preds, start=1):
        print(f"  {rank}. {label} ({prob:.2%})")


if __name__ == "__main__":
    main()
