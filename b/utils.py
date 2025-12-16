import numpy as np
import librosa


def preprocess_audio_to_cqt(audio_path, sr=22050, hop_length=512, n_bins=192, bins_per_octave=24):
    y, _ = librosa.load(audio_path, sr=sr)
    y = librosa.util.normalize(y)
    
    spec = np.abs(librosa.cqt(
        y, 
        sr=sr, 
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    ))
    
    spec = np.log(spec + 1e-10)
    return np.swapaxes(spec, 0, 1)


def midi_to_fret(midi_note, string_num):
    string_open_midi = [40, 45, 50, 55, 59, 64]
    if string_num < 0 or string_num >= 6:
        return 0
    fret = midi_note - string_open_midi[string_num]
    return max(0, min(fret, 24))
