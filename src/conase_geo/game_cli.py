from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
from typing import Optional

from conase_geo.predict import GeoPredictor


def _record_from_microphone(duration: float, sample_rate: int = 16_000) -> Path:
    import sounddevice as sd  # type: ignore
    import soundfile as sf  # type: ignore

    n_samples = int(duration * sample_rate)
    recording = sd.rec(n_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()

    out_path = Path(tempfile.gettempdir()) / f"conase_recording_{int(time.time())}.wav"
    sf.write(out_path, recording, sample_rate)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play the dialect location guesser game from terminal.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--record_seconds", type=float, default=8.0)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _prompt_audio_path(record_seconds: float) -> Optional[Path]:
    print("Choose input:")
    print("1) Provide an audio file path")
    print("2) Record from microphone (requires sounddevice + soundfile)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        try:
            path = _record_from_microphone(duration=record_seconds)
            print(f"Recorded audio saved to: {path}")
            return path
        except Exception as exc:
            print(f"Microphone capture failed ({exc}). Falling back to file path mode.")

    raw_path = input("Enter path to audio file: ").strip().strip('"')
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.exists():
        print(f"File not found: {path}")
        return None
    return path


def main() -> None:
    args = build_arg_parser().parse_args()
    predictor = GeoPredictor(checkpoint_path=args.checkpoint, device=args.device)

    audio_path = _prompt_audio_path(record_seconds=args.record_seconds)
    if audio_path is None:
        print("No valid audio input was provided.")
        return

    preds = predictor.predict(audio_path=audio_path, top_k=args.top_k)
    if not preds:
        print("No predictions available.")
        return

    best_label, best_prob = preds[0]
    print("\nGuess:")
    print(f"  {best_label} ({best_prob:.2%})")

    print("\nTop predictions:")
    for rank, (label, prob) in enumerate(preds, start=1):
        print(f"  {rank}. {label} ({prob:.2%})")


if __name__ == "__main__":
    main()
