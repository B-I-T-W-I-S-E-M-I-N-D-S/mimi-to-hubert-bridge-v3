"""
preprocess_emotion.py — Pre-extract & Cache Features for Emotion Dataset
=========================================================================
Pre-warms the feature cache (Mimi tokens, HuBERT features, prosody) for
all audio files listed in mead_labels.csv. This avoids slow on-the-fly
extraction during the first training epoch.

Usage (single GPU):
    python preprocess_emotion.py --config config.yaml --device cuda

Usage (multi-GPU — parallel extraction, much faster):
    torchrun --nproc_per_node=5 preprocess_emotion.py --config config.yaml --device cuda
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from emotion_dataset import load_emotion_csv, EmotionMimiHuBERTDataset

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract features for MEAD emotion dataset"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device", default="cuda", help="Device for feature extraction")
    args = parser.parse_args()

    # ── DDP setup (when launched with torchrun) ───────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    global_rank = int(os.environ.get("RANK",        0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))

    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    is_main = (global_rank == 0)

    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank{global_rank}] %(levelname)s  %(message)s",
        handlers=handlers,
    )

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg["data"]["emotion_csv"]
    if is_main:
        logger.info(f"Loading emotion CSV: {csv_path}")

    all_samples = load_emotion_csv(csv_path)
    total = len(all_samples)

    # ── Shard across GPUs ─────────────────────────────────────────────────────
    # Each rank processes a non-overlapping slice of the dataset
    shard_size = (total + world_size - 1) // world_size
    start_idx = global_rank * shard_size
    end_idx   = min(start_idx + shard_size, total)
    my_samples = all_samples[start_idx:end_idx]

    logger.info(
        f"Rank {global_rank}: processing samples {start_idx}..{end_idx} "
        f"({len(my_samples)} / {total} total)"
    )

    # ── Build dataset (triggers lazy extractor loading on first access) ───────
    ds = EmotionMimiHuBERTDataset(my_samples, cfg, split="preprocess", device=device)

    # ── Extract all features (populates cache) ────────────────────────────────
    t0 = time.time()
    errors = 0
    for i in range(len(ds)):
        try:
            _ = ds[i]  # triggers extraction + caching
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  Error on sample {start_idx + i}: {e}")
            elif errors == 6:
                logger.warning("  (suppressing further errors...)")

        if (i + 1) % 100 == 0 or (i + 1) == len(ds):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(ds) - i - 1) / max(rate, 0.01)
            logger.info(
                f"  [{i+1}/{len(ds)}]  {rate:.1f} samples/s  "
                f"ETA {eta:.0f}s  errors={errors}"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Rank {global_rank} done: {len(ds)} samples in {elapsed:.1f}s "
        f"({len(ds)/max(elapsed,0.01):.1f} samples/s, {errors} errors)"
    )

    # ── Sync all ranks ────────────────────────────────────────────────────────
    if world_size > 1:
        torch.distributed.barrier()

    if is_main:
        cache_dir = cfg["data"].get("cache_dir", "data/cache")
        n_cached = len(list(Path(cache_dir).glob("*.pt")))
        logger.info(f"\nPreprocessing complete!  {n_cached} cached files in {cache_dir}/")
        logger.info("You can now run training:")
        logger.info("  torchrun --nproc_per_node=5 train.py --config config.yaml")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
