from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess

LOGGER = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_clip(
    input_audio: str | Path,
    output_wav: str | Path,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16_000,
    mono: bool = True,
    overwrite: bool = False,
) -> bool:
    input_path = Path(input_audio)
    output_path = Path(output_wav)
    if not input_path.exists():
        LOGGER.warning("Input audio missing: %s", input_path)
        return False
    if output_path.exists() and not overwrite:
        return True
    if end_sec <= start_sec:
        return False
    if not ffmpeg_available():
        LOGGER.error("ffmpeg not found on PATH.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    channels = "1" if mono else "2"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{max(0.0, start_sec):.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-ac",
        channels,
        "-ar",
        str(sample_rate),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.warning("ffmpeg clip failed for %s: %s", input_path, result.stderr.strip())
        return False
    return output_path.exists()
