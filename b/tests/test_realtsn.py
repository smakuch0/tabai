import torch
import numpy as np
from b.TSN import TSN, RealTSN

print("Testing RealTSN architecture...")

model = RealTSN(context_window=9)
model.eval()

batch_size = 4
dummy_input = torch.randn(batch_size, 1, 192, 9)

with torch.no_grad():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, 6, 25)")

assert output.shape == (batch_size, 6, 25), f"Wrong output shape: {output.shape}"

num_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {num_params:,}")

print("\nTesting with TSN wrapper...")
tsn = TSN()
tsn.build_model()

print(f"Device: {tsn.device}")
print(f"Model type: {type(tsn.model).__name__}")

dummy_spec = np.random.randn(50, 192)
predictions = tsn.predict(dummy_spec, context_window=9)

print(f"\nPrediction shape: {predictions.shape}")
print(f"Prediction dtype: {predictions.dtype}")
print(f"Prediction range: [{predictions.min()}, {predictions.max()}]")

assert predictions.shape == (50, 6), f"Wrong prediction shape: {predictions.shape}"
assert predictions.min() >= 0 and predictions.max() <= 24, "Predictions out of range"
