# CoNASE Dialect Location Guesser

Toy ML repo for a "guess your hometown/city" style game using CoNASE-style metadata + speech clips.

## Important Ethics, Legal, and Safety Notes
- This is a toy game. Predictions are probabilistic and often wrong.
- Do not use this to identify people or infer private traits about non-consenting speakers.
- Respect CoNASE dataset terms: non-commercial usage and no redistribution.
- Audio retrieval is optional and pluggable. If you fetch audio from public platforms, you are responsible for complying with platform terms, copyright, and speaker rights.
- Prefer `--provider local` with audio you own or are explicitly allowed to process.

## What This Repo Supports
- Labeling modes:
  - `state` (default and recommended baseline)
  - `location_topk` (restrict to top-K frequent locations)
  - `geo_grid` (discretized lat/lon cells)
- Models:
  - `timing_only`: MLP on token timing features (no audio needed)
  - `audio_only`: frozen `wav2vec2_base` (or MFCC fallback) + classifier
  - `audio_plus_timing`: audio embedding + timing + prosody features
  - `mpsa_densenet`: MPSA-DenseNet-style MFCC model (DenseNet with PSA modules)
  - `text_hashing_sgd` (new): writing-only geolocation baseline for large CSV corpora
  - `text_hashing_torch_linear` (new): GPU-capable writing-only baseline (`--trainer torch`)
- Group-aware train/val/test split (default `split_by=channel_id`).
- Optional Streamlit app and terminal game CLI.

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

Optional extras:
```bash
pip install -e .[app,youtube,mic,test]
```

System tools:
- `ffmpeg` is required for clipping audio.
- `yt-dlp` is required only for `--provider youtube`.

## 1) Build Manifest (from downloaded CoNASE CSV)
```bash
python -m conase_geo.data.make_manifest ^
  --csv_path ./data/conase_pos.csv ^
  --out_manifest ./data/manifest.csv ^
  --label_type state ^
  --clip_seconds 10 ^
  --clips_per_video 2 ^
  --max_videos 2000 ^
  --min_label_examples 200
```

Output manifest columns include:
`video_id, channel_id, country, state, location, lat, lon, clip_start, clip_end, label, text_window, token_times_json`

`token_times_json` is a JSON list of token timestamps in the clip window (best effort, relative to clip start).

## 2) Optional Audio Fetch + Clip Preparation (OFF by default workflow)
```bash
python -m conase_geo.data.audio_fetch ^
  --manifest ./data/manifest.csv ^
  --audio_cache_dir ./data/audio_cache ^
  --clips_dir ./data/clips ^
  --provider youtube ^
  --max_workers 4
```

This writes `manifest_with_audio.csv` (adds `audio_path`).

Resumable behavior:
- Existing raw audio/cache files are reused.
- Existing clip files are skipped.
- Failed downloads/clips are retried.

Local provider example:
```bash
python -m conase_geo.data.audio_fetch ^
  --manifest ./data/manifest.csv ^
  --audio_cache_dir ./data/audio_cache ^
  --clips_dir ./data/clips ^
  --provider local ^
  --local_audio_template "./data/audio_cache/{video_id}.wav"
```

## 3) Train
```bash
python -m conase_geo.train ^
  --manifest ./data/manifest_with_audio.csv ^
  --label_type state ^
  --model audio_plus_timing ^
  --encoder wav2vec2_base ^
  --batch_size 8 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --output_dir ./checkpoints ^
  --split_by channel_id
```

MPSA-DenseNet run:
```bash
python -m conase_geo.train ^
  --manifest ./data/manifest_with_audio.csv ^
  --label_type state ^
  --model mpsa_densenet ^
  --batch_size 8 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --mpsa_input_seconds 6 ^
  --mpsa_n_mfcc 64 ^
  --mpsa_block_config 6,12,24,16 ^
  --output_dir ./checkpoints_mpsa ^
  --split_by channel_id
```

Training reports:
- Accuracy
- Top-3 accuracy

Artifacts:
- `checkpoints/best.pt`
- `checkpoints/label_mapping.json`
- `checkpoints/manifest_with_splits.csv`
- `checkpoints/metrics.json`

## Minimal Demo (No Audio Fetch)
Use only token timing features:
```bash
python -m conase_geo.train ^
  --manifest ./data/manifest.csv ^
  --label_type state ^
  --model timing_only ^
  --batch_size 32 ^
  --epochs 5 ^
  --lr 1e-3 ^
  --output_dir ./checkpoints_timing ^
  --split_by channel_id
```

## Writing-Only Mode (No Timing/Audio Needed)
Train a text geolocation model directly from large CSV files that contain writing + location labels.

Example on your distributable file:
```bash
python -m conase_geo.train_text ^
  --csv_path "C:/Users/jklus/Downloads/APCSA Code/accent_guesser_thing/conase_distributable_a.csv/conase_distributable_a.csv" ^
  --output_dir ./checkpoints_text ^
  --sep "|" ^
  --text_col text ^
  --label_col state ^
  --group_col channel_id ^
  --min_label_examples 200 ^
  --epochs 3 ^
  --chunksize 5000 ^
  --batch_size 2048
```

Use all labels (no top-k cap):
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text_all_labels ^
  --label_col state ^
  --top_k_labels 0 ^
  --min_label_examples 1
```

Autosave during training (recommended for long runs on laptops):
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text ^
  --autosave_every_batches 0 ^
  --autosave_every_minutes 30
```

This writes a rolling checkpoint at:
- `./checkpoints_text/text_model_autosave.joblib`

If training is interrupted, resume from autosave:
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text ^
  --resume_checkpoint ./checkpoints_text/text_model_autosave.joblib
```

GPU trainer (faster if CUDA is available):
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text_gpu ^
  --trainer torch ^
  --device auto ^
  --torch_optimizer sgd ^
  --torch_lr 0.05 ^
  --batch_size 4096 ^
  --epochs 3
```

Geo-aware loss (optional, torch trainer):
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text_geo ^
  --trainer torch ^
  --geo_loss_mode centroid ^
  --geo_loss_mix 0.35 ^
  --geo_sigma_km 850 ^
  --lat_col lat ^
  --lon_col lon ^
  --latlong_col latlong
```
This mixes hard class cross-entropy with a distance-weighted soft target over labels.

Speed tips for large CSV:
- Increase `--batch_size` (try `4096`, `8192`, `16384` depending on VRAM).
- Increase `--chunksize` (for example `8000` to `20000`).
- Use `--ngram_max 4` instead of `5` for a substantial speedup.
- Set `--n_features 262144` instead of `524288` if throughput is more important than quality.
- Use `--max_train_minutes` to cap wall-clock runtime, and `--skip_eval` to spend more time on training updates.

Per-epoch validation + training history:
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text_history ^
  --trainer torch ^
  --eval_every_epoch ^
  --epoch_history_path ./checkpoints_text_history/epoch_history.jsonl
```
This writes:
- `epoch_history.jsonl` (append-only per-epoch records)
- `epoch_history.csv`
- `epoch_history.json`

8-hour maximize-throughput run (all labels, GPU+CPU):
```bash
python -m conase_geo.train_text ^
  --csv_path "..." ^
  --output_dir ./checkpoints_text_8h ^
  --sep "|" ^
  --text_col text ^
  --label_col state ^
  --group_col channel_id ^
  --max_rows 0 ^
  --min_text_chars 20 ^
  --min_label_examples 1 ^
  --top_k_labels 0 ^
  --val_fraction 0.1 ^
  --test_fraction 0.1 ^
  --seed 42 ^
  --analyzer char_wb ^
  --ngram_min 3 ^
  --ngram_max 4 ^
  --n_features 262144 ^
  --trainer torch ^
  --device auto ^
  --torch_optimizer sgd ^
  --torch_lr 0.05 ^
  --torch_weight_decay 1e-6 ^
  --epochs 999 ^
  --chunksize 12000 ^
  --batch_size 8192 ^
  --autosave_every_batches 0 ^
  --autosave_every_minutes 30 ^
  --max_train_minutes 480 ^
  --skip_eval
```

This is designed for very large files:
- chunked streaming CSV reads
- group-aware split by `channel_id` (to reduce leakage)
- no full in-memory text matrix required

Predict from writing:
```bash
python -m conase_geo.predict_text ^
  --checkpoint ./checkpoints_text/text_model.joblib ^
  --text "Y'all fixin' to head over after supper, maybe around six." ^
  --top_k 5
```

Interactive writing game:
```bash
python -m conase_geo.game_text_cli ^
  --checkpoint ./checkpoints_text/text_model.joblib ^
  --top_k 5
```

## 4) Predict
```bash
python -m conase_geo.predict ^
  --checkpoint ./checkpoints/best.pt ^
  --audio ./my_sample.wav ^
  --top_k 5
```

If your checkpoint is `timing_only`, pass token timing JSON for useful predictions:
```bash
python -m conase_geo.predict ^
  --checkpoint ./checkpoints_timing/best.pt ^
  --audio ./my_sample.wav ^
  --token_times_json "[0.12, 0.47, 0.81, 1.33]" ^
  --top_k 5
```

## 5) Game CLI
```bash
python -m conase_geo.game_cli ^
  --checkpoint ./checkpoints/best.pt ^
  --top_k 5
```

Game flow:
- choose local audio file path, or
- record from microphone (if `sounddevice` is installed).

## 6) Optional Streamlit App
```bash
streamlit run app_streamlit.py
```

## Notes on CPU Practicality
- Defaults are set for small subset runs (`max_videos=2000`, `clips_per_video=2`, `clip_seconds=10`).
- `wav2vec2_base` runs on CPU but can be slow; use `--encoder mfcc` for a faster fallback baseline.

## Tests
```bash
pytest -q
```
