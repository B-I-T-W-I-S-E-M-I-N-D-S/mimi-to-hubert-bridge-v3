"""
emotion_dataset.py — Emotion-Aware Data Loading
=================================================
CSV-based dataset for the MEAD emotion recognition extension.

Reads:
  - mead_labels.csv  (columns: filename, emotion)
  - Audio .wav files from the audio directory

Returns all fields from the original MimiHuBERTDataset PLUS:
  - emotion_label : int64 scalar  — index into EMOTION_CLASSES

Includes:
  - Stratified train/val split (by emotion)
  - Per-class sample count → inverse-frequency weights for loss
  - Custom collate_fn that extends the original with emotion labels
"""

import csv
import logging
import math
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Emotion label mapping — 8 MEAD emotions
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_CLASSES = [
    "angry",
    "contempt",
    "disgusted",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTION_CLASSES)}

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (guarded)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False
    logger.warning("torchaudio not found — audio loading will fail at runtime.")

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("sklearn not found — using manual stratified split fallback.")


# ──────────────────────────────────────────────────────────────────────────────
# CSV Parsing & Train/Val Split
# ──────────────────────────────────────────────────────────────────────────────

def load_emotion_csv(csv_path: str) -> List[Dict]:
    """
    Read mead_labels.csv and return a list of dicts:
      [{"filename": "...", "emotion": "happy", "emotion_idx": 4}, ...]

    Skips rows with unknown emotion labels and logs warnings.
    """
    samples = []
    unknown_labels = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            emotion = row["emotion"].strip().lower()

            if emotion not in EMOTION_TO_IDX:
                unknown_labels.add(emotion)
                continue

            samples.append({
                "filename": filename,
                "emotion": emotion,
                "emotion_idx": EMOTION_TO_IDX[emotion],
            })

    if unknown_labels:
        logger.warning(
            f"[EmotionDataset] Skipped {len(unknown_labels)} unknown emotion labels: "
            f"{sorted(unknown_labels)}"
        )

    logger.info(f"[EmotionDataset] Loaded {len(samples)} samples from {csv_path}")
    return samples


def stratified_split(
    samples: List[Dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train/val sets, stratified by emotion label.
    Uses sklearn if available, otherwise a manual per-class split.
    """
    if SKLEARN_OK:
        labels = [s["emotion_idx"] for s in samples]
        train_samples, val_samples = train_test_split(
            samples, test_size=val_ratio, random_state=seed,
            stratify=labels,
        )
        logger.info(
            f"[EmotionDataset] Split: {len(train_samples)} train / "
            f"{len(val_samples)} val  (stratified, sklearn)"
        )
        return train_samples, val_samples

    # Manual stratified split fallback
    from collections import defaultdict
    import random

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s["emotion_idx"]].append(s)

    train_samples, val_samples = [], []
    for cls_idx, cls_samples in by_class.items():
        rng.shuffle(cls_samples)
        n_val = max(1, int(len(cls_samples) * val_ratio))
        val_samples.extend(cls_samples[:n_val])
        train_samples.extend(cls_samples[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    logger.info(
        f"[EmotionDataset] Split: {len(train_samples)} train / "
        f"{len(val_samples)} val  (stratified, manual)"
    )
    return train_samples, val_samples


def compute_class_counts(samples: List[Dict], num_classes: int) -> torch.Tensor:
    """Compute per-class sample counts for class weight calculation."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for s in samples:
        counts[s["emotion_idx"]] += 1
    for i, name in enumerate(EMOTION_CLASSES[:num_classes]):
        logger.info(f"  Class {i} ({name:>12s}): {counts[i].item():5d} samples")
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class EmotionMimiHuBERTDataset(Dataset):
    """
    Paired dataset for emotion-aware bridge training.

    **Performance strategy — two modes:**

    1. **Preloaded (fast):** If all samples have cached features on disk
       (from running preprocess_emotion.py), everything is loaded into RAM
       at __init__ time. __getitem__ is a pure dict-lookup → zero disk I/O
       during training. Epochs drop from ~11 min to ~1 min.

    2. **On-the-fly (slow fallback):** If cache is incomplete, features
       are extracted on-the-fly and cached for next epoch.
    """

    def __init__(
        self,
        samples: List[Dict],
        cfg: dict,
        split: str = "train",
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.split = split
        self.samples = samples
        self.sr = cfg["data"]["sample_rate"]                    # 16000
        self.mimi_sr = cfg["data"].get("mimi_sample_rate", 24000)
        self.max_len = int(cfg["data"]["max_audio_seconds"] * self.sr)
        self.cache_features = cfg["data"].get("cache_features", True)
        self.cache_dir = Path(cfg["data"].get("cache_dir", "data/cache"))
        self.hop_length = cfg["training"].get("hop_length", 160)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-init extractors (only used in on-the-fly fallback mode)
        self._mimi = None
        self._hubert = None
        self._device = device

        # ── Try to preload all features into RAM ─────────────────────────
        self._preloaded_data = None  # List[dict] when preloaded
        if self.cache_features and split != "preprocess":
            self._try_preload_all()

    # ─────────────────────────────────────────────────────────────────────
    # RAM preloading — eliminates ALL disk I/O during training
    # ─────────────────────────────────────────────────────────────────────

    def _cache_path(self, audio_path: str, suffix: str) -> Path:
        h = hashlib.md5(audio_path.encode()).hexdigest()
        return self.cache_dir / f"{h}_{suffix}.pt"

    def _try_preload_all(self):
        """
        Load ALL cached features (tokens, hubert, prosody) into RAM.
        If any sample is missing from cache, skip it.
        """
        import time
        t0 = time.time()
        n = len(self.samples)
        logger.info(
            f"[{self.split}] Preloading {n} samples from cache into RAM..."
        )

        data = []
        valid_samples = []
        missing = 0

        for i, sample in enumerate(self.samples):
            audio_path = sample["filename"]
            emotion_idx = sample["emotion_idx"]

            # Check cache existence for required features
            mimi_cp   = self._cache_path(audio_path, "mimi")
            hubert_cp = self._cache_path(audio_path, "hubert")

            if not mimi_cp.exists() or not hubert_cp.exists():
                missing += 1
                if missing <= 3:
                    logger.warning(f"  Cache miss: {audio_path}")
                elif missing == 4:
                    logger.warning("  (suppressing further cache-miss warnings...)")
                continue

            # Load from cache — this only happens ONCE at init
            try:
                tokens = torch.load(mimi_cp, map_location="cpu", weights_only=False)
                hubert = torch.load(hubert_cp, map_location="cpu", weights_only=False)
            except Exception as e:
                missing += 1
                if missing <= 3:
                    logger.warning(f"  Corrupt cache for {audio_path}: {e}")
                continue

            # Enforce 2:1 ratio (tokens → hubert frames)
            T_m = tokens.shape[0]
            T_h = hubert.shape[0]
            T_min = min(T_m, T_h // 2)
            tokens = tokens[:T_min]
            hubert = hubert[:T_min * 2]
            T_h = T_min * 2

            # Try loading cached prosody (optional — zeros fallback is safe)
            prosody_cp = self._cache_path(audio_path + f"_L{T_h}", "prosody")
            f0 = torch.zeros(T_h)
            energy = torch.zeros(T_h)
            voiced = torch.zeros(T_h, dtype=torch.bool)

            if prosody_cp.exists():
                try:
                    prosody = torch.load(prosody_cp, map_location="cpu", weights_only=False)
                    if isinstance(prosody, tuple) and len(prosody) == 3:
                        pf0, penergy, pvoiced = prosody
                        # Ensure correct lengths
                        if pf0.shape[0] == T_h:
                            f0, energy, voiced = pf0, penergy, pvoiced
                        else:
                            f0 = F.interpolate(
                                pf0.unsqueeze(0).unsqueeze(0).float(),
                                size=T_h, mode="linear", align_corners=False,
                            ).squeeze(0).squeeze(0)
                            energy = F.interpolate(
                                penergy.unsqueeze(0).unsqueeze(0).float(),
                                size=T_h, mode="linear", align_corners=False,
                            ).squeeze(0).squeeze(0)
                            voiced = F.interpolate(
                                pvoiced.unsqueeze(0).unsqueeze(0).float(),
                                size=T_h, mode="linear", align_corners=False,
                            ).squeeze(0).squeeze(0) > 0.5
                except Exception:
                    pass  # keep zeros — prosody loss still works

            data.append({
                "tokens":        tokens,
                "hubert":        hubert,
                "f0":            f0,
                "energy":        energy,
                "voiced":        voiced,
                "phone_labels":  None,
                "emotion_label": torch.tensor(emotion_idx, dtype=torch.long),
                "audio_path":    audio_path,
            })
            valid_samples.append(sample)

            # Progress logging every 5000 samples
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                logger.info(f"  [{i+1}/{n}] preloaded in {elapsed:.1f}s...")

        elapsed = time.time() - t0

        if missing > 0:
            logger.warning(
                f"  {missing}/{n} samples missing from cache. "
                f"Run preprocess_emotion.py first for full preloading."
            )

        if len(data) == 0:
            logger.warning(
                f"[{self.split}] No cached data found — falling back to "
                f"on-the-fly extraction (slow). Run preprocess_emotion.py first!"
            )
            return

        self._preloaded_data = data
        self.samples = valid_samples
        logger.info(
            f"[{self.split}] Preloaded {len(data)} samples into RAM "
            f"in {elapsed:.1f}s — training will have ZERO disk I/O"
        )

    # ─────────────────────────────────────────────────────────────────────
    # On-the-fly extraction helpers (fallback when cache is incomplete)
    # ─────────────────────────────────────────────────────────────────────

    def _get_mimi(self):
        if self._mimi is None:
            from dataset import MimiExtractor
            self._mimi = MimiExtractor(self.cfg["paths"]["mimi_model"], self._device)
        return self._mimi

    def _get_hubert(self):
        if self._hubert is None:
            from dataset import HuBERTExtractor
            self._hubert = HuBERTExtractor(self.cfg["paths"]["hubert_model"], self._device)
        return self._hubert

    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        max_native = int(self.cfg["data"]["max_audio_seconds"] * native_sr)
        if waveform.shape[-1] > max_native:
            waveform = waveform[:, :max_native]
        return waveform, native_sr

    def _get_or_cache(self, audio_path: str, key: str, extractor_fn):
        cp = self._cache_path(audio_path, key)
        if self.cache_features and cp.exists():
            try:
                return torch.load(cp, map_location="cpu", weights_only=False)
            except Exception:
                cp.unlink(missing_ok=True)
        result = extractor_fn()
        if self.cache_features:
            torch.save(result, cp)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Core __getitem__
    # ─────────────────────────────────────────────────────────────────────

    def __len__(self):
        if self._preloaded_data is not None:
            return len(self._preloaded_data)
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # ── Fast path: return from RAM (zero disk I/O) ────────────────
        if self._preloaded_data is not None:
            return self._preloaded_data[idx]

        # ── Slow path: on-the-fly extraction (fallback) ───────────────
        return self._extract_on_the_fly(idx)

    def _extract_on_the_fly(self, idx: int) -> dict:
        """Fallback: extract features from audio (slow, caches results)."""
        sample = self.samples[idx]
        audio_path = sample["filename"]
        emotion_idx = sample["emotion_idx"]

        wav, native_sr = self._load_audio(audio_path)

        if native_sr != self.sr and TORCHAUDIO_OK:
            wav_16k = torchaudio.functional.resample(wav, native_sr, self.sr)
        else:
            wav_16k = wav
        wav_np = wav_16k.squeeze().numpy()

        tokens = self._get_or_cache(
            audio_path, "mimi",
            lambda: self._get_mimi().extract(wav, native_sr)
        )
        hubert = self._get_or_cache(
            audio_path, "hubert",
            lambda: self._get_hubert().extract(wav, native_sr),
        )

        T_m = tokens.shape[0]
        T_h = hubert.shape[0]
        T_min = min(T_m, T_h // 2)
        tokens = tokens[:T_min]
        hubert = hubert[:T_min * 2]
        T_h = T_min * 2

        from dataset import extract_f0_energy

        def _extract_prosody():
            f0_np, energy_np, voiced_np = extract_f0_energy(
                wav_np, self.sr, self.hop_length
            )
            f0_r = torch.from_numpy(self._resample_array(f0_np, T_h))
            energy_r = torch.from_numpy(self._resample_array(energy_np, T_h))
            voiced_r = torch.from_numpy(
                self._resample_array(voiced_np.astype(np.float32), T_h) > 0.5
            )
            return (f0_r, energy_r, voiced_r)

        prosody = self._get_or_cache(
            audio_path + f"_L{T_h}", "prosody", _extract_prosody,
        )
        if isinstance(prosody, tuple) and len(prosody) == 3:
            f0, energy, voiced = prosody
        else:
            f0, energy, voiced = _extract_prosody()

        return {
            "tokens":        tokens,
            "hubert":        hubert,
            "f0":            f0,
            "energy":        energy,
            "voiced":        voiced,
            "phone_labels":  None,
            "emotion_label": torch.tensor(emotion_idx, dtype=torch.long),
            "audio_path":    audio_path,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        indices = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(indices, np.arange(len(arr)), arr).astype(arr.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Collate Function (extends original with emotion labels)
# ──────────────────────────────────────────────────────────────────────────────

def emotion_collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length sequences and collate emotion labels.
    Returns the same dict as the original collate_fn PLUS:
      emotion_labels : (B,) int64
    """
    # Sort by descending token length
    batch = sorted(batch, key=lambda x: x["tokens"].shape[0], reverse=True)

    max_T_m = max(b["tokens"].shape[0] for b in batch)
    max_T_h = max_T_m * 2

    B = len(batch)
    feat_dim = batch[0]["hubert"].shape[-1]
    num_codebooks = batch[0]["tokens"].shape[-1]

    tokens_out = torch.zeros(B, max_T_m, num_codebooks, dtype=torch.long)
    hubert_out = torch.zeros(B, max_T_h, feat_dim)
    f0_out = torch.zeros(B, max_T_h)
    energy_out = torch.zeros(B, max_T_h)
    voiced_out = torch.zeros(B, max_T_h, dtype=torch.bool)
    mask_out = torch.zeros(B, max_T_h, dtype=torch.bool)
    phone_out = torch.full((B, max_T_h), -100, dtype=torch.long)
    emotion_out = torch.zeros(B, dtype=torch.long)

    token_lengths = []
    for i, sample in enumerate(batch):
        T_m = sample["tokens"].shape[0]
        T_h = T_m * 2
        tokens_out[i, :T_m] = sample["tokens"]
        hubert_out[i, :T_h] = sample["hubert"]
        f0_out[i, :T_h] = sample["f0"]
        energy_out[i, :T_h] = sample["energy"]
        voiced_out[i, :T_h] = sample["voiced"]
        mask_out[i, :T_h] = True
        if sample["phone_labels"] is not None:
            phone_out[i, :T_h] = sample["phone_labels"]
        emotion_out[i] = sample["emotion_label"]
        token_lengths.append(T_m)

    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    frame_lengths = token_lengths * 2

    return {
        "tokens":         tokens_out,
        "hubert":         hubert_out,
        "f0":             f0_out,
        "energy":         energy_out,
        "voiced_mask":    voiced_out,
        "mask":           mask_out,
        "phone_labels":   phone_out,
        "input_lengths":  frame_lengths,
        "ctc_targets":    None,
        "target_lengths": None,
        "emotion_labels": emotion_out,       # (B,) int64
    }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_emotion_dataloaders(
    cfg: dict,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build train/val DataLoaders from mead_labels.csv.

    Returns:
        train_loader : DataLoader
        val_loader   : DataLoader
        class_counts : (num_emotions,) tensor — for setting class weights in loss
    """
    csv_path = cfg["data"]["emotion_csv"]
    val_ratio = cfg["data"].get("emotion_val_ratio", 0.1)
    seed = cfg["training"].get("seed", 42)
    num_emotions = cfg["model"].get("num_emotions", 8)

    # Load and split
    all_samples = load_emotion_csv(csv_path)
    train_samples, val_samples = stratified_split(all_samples, val_ratio, seed)

    # Class distribution
    class_counts = compute_class_counts(train_samples, num_emotions)

    # Build datasets
    train_ds = EmotionMimiHuBERTDataset(train_samples, cfg, "train", device)
    val_ds = EmotionMimiHuBERTDataset(val_samples, cfg, "val", device)

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    num_workers = d_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=emotion_collate_fn,
        pin_memory=(device != "cpu"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=emotion_collate_fn,
        pin_memory=(device != "cpu"),
    )

    logger.info(
        f"[EmotionDataset] Loaders ready: "
        f"{len(train_ds)} train / {len(val_ds)} val samples, "
        f"batch_size={t_cfg['batch_size']}"
    )

    return train_loader, val_loader, class_counts

