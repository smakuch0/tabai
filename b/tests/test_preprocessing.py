import numpy as np
import os
from b.TSN import TSN

sr = 22050
duration = 2
t = np.linspace(0, duration, int(sr * duration))
audio = 0.3 * np.sin(2 * np.pi * 440 * t)

test_file = str(os.path.dirname(os.path.abspath(__file__))) + '/test_audio.wav'

print("Testing TSN preprocessing...")

tsn = TSN()
tsn.build_model()

audio_repr = tsn.preprocess_audio(test_file)
print(f"Preprocessed shape: {audio_repr.shape}")
print(f"Expected: ({audio_repr.shape[0]}, {tsn.n_bins})")

predictions = tsn.predict(audio_repr, context_window=9)
print(f"Predictions shape: {predictions.shape}")
print(f"Expected: ({audio_repr.shape[0]}, {tsn.num_strings})")

print("\nCQT parameters:")
print(f"  sr: {tsn.sr}")
print(f"  hop_length: {tsn.hop_length}")
print(f"  n_bins: {tsn.n_bins}")
print(f"  bins_per_octave: {tsn.bins_per_octave}")
