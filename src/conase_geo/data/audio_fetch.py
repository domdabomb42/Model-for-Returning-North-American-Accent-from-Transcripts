from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from conase_geo.data.clip_audio import extract_clip, ffmpeg_available

LOGGER = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            self._last = time.monotonic()


def _download_youtube_audio(
    video_id: str,
    out_wav: Path,
    retries: int,
    limiter: RateLimiter,
) -> bool:
    if out_wav.exists():
        return True
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = str(out_wav.with_suffix(".%(ext)s"))

    for attempt in range(1, retries + 1):
        limiter.wait()
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "-o",
            output_template,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_wav.exists():
            return True

        fallback = sorted(out_wav.parent.glob(f"{video_id}.*"))
        if fallback:
            for candidate in fallback:
                if candidate.suffix.lower() == ".wav":
                    candidate.rename(out_wav)
                    return True

        LOGGER.warning(
            "yt-dlp failed for %s (attempt %s/%s): %s",
            video_id,
            attempt,
            retries,
            result.stderr.strip(),
        )
        time.sleep(1.5 * attempt)
    return out_wav.exists()


def _process_video_download(
    video_id: str,
    provider: str,
    audio_cache_dir: Path,
    local_audio_template: Optional[str],
    retries: int,
    limiter: RateLimiter,
) -> Tuple[str, Optional[Path]]:
    if provider == "youtube":
        target = audio_cache_dir / f"{video_id}.wav"
        ok = _download_youtube_audio(video_id=video_id, out_wav=target, retries=retries, limiter=limiter)
        return video_id, target if ok else None

    if provider == "local":
        if not local_audio_template:
            raise ValueError("provider=local requires --local_audio_template.")
        candidate = Path(local_audio_template.format(video_id=video_id))
        return video_id, candidate if candidate.exists() else None

    raise ValueError(f"Unsupported provider: {provider}")


def _clip_from_manifest_row(row: pd.Series, raw_audio_path: Path, clips_dir: Path) -> Optional[Path]:
    video_id = str(row["video_id"]).strip()
    try:
        start = float(row["clip_start"])
        end = float(row["clip_end"])
    except (TypeError, ValueError):
        return None

    clip_name = f"{video_id}_{int(start * 1000)}_{int(end * 1000)}.wav"
    clip_path = clips_dir / clip_name
    ok = extract_clip(raw_audio_path, clip_path, start_sec=start, end_sec=end, sample_rate=16_000, mono=True)
    return clip_path if ok else None


def build_manifest_with_audio(
    manifest_path: Path,
    audio_cache_dir: Path,
    clips_dir: Path,
    provider: str = "local",
    local_audio_template: Optional[str] = None,
    max_workers: int = 4,
    retries: int = 3,
    rate_limit_seconds: float = 1.0,
    out_manifest: Optional[Path] = None,
) -> pd.DataFrame:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required but not found on PATH.")
    if provider == "local" and not local_audio_template:
        raise ValueError("provider=local requires --local_audio_template, e.g. ./data/audio_cache/{video_id}.wav")
    if provider == "youtube" and shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is required for provider=youtube.")

    LOGGER.warning(
        "Audio retrieval is optional. Respect dataset terms, platform terms, and speaker rights. "
        "Use provider=local when possible."
    )

    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    for col in ["video_id", "clip_start", "clip_end"]:
        if col not in df.columns:
            raise ValueError(f"Missing required manifest column: {col}")

    video_ids = sorted(df["video_id"].astype(str).str.strip().unique().tolist())
    limiter = RateLimiter(min_interval_seconds=rate_limit_seconds)
    raw_audio_map: Dict[str, Path] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_video_download,
                video_id,
                provider,
                audio_cache_dir,
                local_audio_template,
                retries,
                limiter,
            )
            for video_id in video_ids
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preparing raw audio"):
            video_id, path = future.result()
            if path is not None and path.exists():
                raw_audio_map[video_id] = path

    audio_paths: Dict[int, str] = {}

    def _row_task(idx: int, row: pd.Series) -> Tuple[int, str]:
        video_id = str(row["video_id"]).strip()
        raw_path = raw_audio_map.get(video_id)
        if raw_path is None:
            return idx, ""
        clip_path = _clip_from_manifest_row(row=row, raw_audio_path=raw_path, clips_dir=clips_dir)
        return idx, str(clip_path) if clip_path else ""

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_row_task, idx, row) for idx, row in df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting clips"):
            idx, audio_path = future.result()
            audio_paths[idx] = audio_path

    df["audio_path"] = [audio_paths.get(i, "") for i in range(len(df))]

    if out_manifest is None:
        out_manifest = manifest_path.with_name(f"{manifest_path.stem}_with_audio.csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_manifest, index=False)

    found = int((df["audio_path"].astype(str).str.len() > 0).sum())
    LOGGER.info("Saved %s with %s/%s rows containing audio_path.", out_manifest, found, len(df))
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optional audio preparation for CoNASE clips. "
            "Respect dataset/platform terms and legal rights for audio retrieval."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio_cache_dir", required=True, type=Path)
    parser.add_argument("--clips_dir", required=True, type=Path)
    parser.add_argument("--provider", choices=["youtube", "local"], default="local")
    parser.add_argument(
        "--local_audio_template",
        type=str,
        default=None,
        help="Template for provider=local, e.g. ./data/audio_cache/{video_id}.wav",
    )
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--rate_limit_seconds", type=float, default=1.0)
    parser.add_argument("--out_manifest", type=Path, default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()
    build_manifest_with_audio(
        manifest_path=args.manifest,
        audio_cache_dir=args.audio_cache_dir,
        clips_dir=args.clips_dir,
        provider=args.provider,
        local_audio_template=args.local_audio_template,
        max_workers=args.max_workers,
        retries=args.retries,
        rate_limit_seconds=args.rate_limit_seconds,
        out_manifest=args.out_manifest,
    )


if __name__ == "__main__":
    main()
