"""Quick smoke test for the emotion-aware bridge module."""
import torch
import yaml

# Load config
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

print('=== Model Smoke Test ===')
from model import MimiHuBERTBridge
model = MimiHuBERTBridge(cfg)
p = model.get_param_count()
print(f'  Parameters: {p["trainable"]:,} trainable / {p["total"]:,} total')
print(f'  Emotion head: {model.emotion_head is not None}')
print(f'  num_emotions: {model.num_emotions}')

# Forward pass
B, T, K = 2, 10, 8
tokens = torch.randint(0, 2048, (B, T, K))
features, kvs, emotion_logits = model(tokens)
print(f'  Input  tokens:  {tokens.shape}')
print(f'  Output features: {features.shape}')
print(f'  Emotion logits:  {emotion_logits.shape if emotion_logits is not None else None}')

# Verify shapes
upsample = cfg['model']['upsample_factor']
assert features.shape == (B, T * upsample, cfg['model']['output_dim']), f'Bad shape: {features.shape}'
assert emotion_logits.shape == (B, cfg['model']['num_emotions']), f'Bad emotion shape: {emotion_logits.shape}'
print('  Shape assertions passed')

print()
print('=== Loss Smoke Test ===')
from losses import BridgeLoss
criterion = BridgeLoss(cfg)
print(f'  Emotion loss enabled: {criterion.emotion is not None}')

target = torch.randn_like(features)
batch = {
    'mask': torch.ones(B, T * upsample, dtype=torch.bool),
    'f0': torch.randn(B, T * upsample),
    'energy': torch.randn(B, T * upsample),
    'voiced_mask': torch.ones(B, T * upsample, dtype=torch.bool),
    'emotion_labels': torch.randint(0, 8, (B,)),
}
total, logs = criterion(features, target, batch, emotion_logits=emotion_logits)
print(f'  Total loss: {total.item():.4f}')
for k, v in logs.items():
    print(f'    {k}: {v}')
print('  Loss computation passed')

print()
print('=== Emotion Dataset Import Test ===')
from emotion_dataset import EMOTION_CLASSES, IDX_TO_EMOTION, EmotionMimiHuBERTDataset
print(f'  Emotion classes: {EMOTION_CLASSES}')
print('  Dataset import passed')

# Test streaming mode (3-tuple return)
print()
print('=== Streaming Smoke Test ===')
model.eval()
with torch.no_grad():
    feats, kvs, emo = model(tokens[:1, :5, :], use_cache=True)
    print(f'  Streaming features: {feats.shape}')
    print(f'  KV cache present: {kvs is not None}')
    print(f'  Emotion (streaming): {emo}')  # should be None during streaming

print()
print('=== All smoke tests PASSED ===')
