import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
import jams

from b.utils import preprocess_audio_to_cqt, midi_to_fret


def load_jams_annotation(jams_path):
    try:
        jam = jams.load(str(jams_path))
    except Exception as e:
        print(f"Error loading JAMS file: {jams_path}")
        print(f"File exists: {jams_path.exists()}")
        print(f"File is file: {jams_path.is_file()}")
        if jams_path.exists():
            print(f"File size: {jams_path.stat().st_size} bytes")
        raise
    
    notes = []
    
    note_midi_anns = [(i, ann) for i, ann in enumerate(jam.annotations) if ann.namespace == 'note_midi']
    
    for string_idx, (ann_idx, ann) in enumerate(note_midi_anns[:6]):
        for obs in ann.data:
            time = obs.time
            duration = obs.duration
            midi = int(round(obs.value))
            
            fret = midi_to_fret(midi, string_idx)
            notes.append((time, duration, midi, string_idx, fret))
    
    return notes


def generate_frame_labels(notes, total_frames, hop_length=512, sr=22050):
    labels = np.zeros((total_frames, 6), dtype=int)
    
    for time, duration, midi, string_num, fret in notes:
        start_frame = librosa.time_to_frames(time, sr=sr, hop_length=hop_length)
        end_frame = librosa.time_to_frames(time + duration, sr=sr, hop_length=hop_length)
        
        for frame in range(start_frame, min(end_frame, total_frames)):
            if frame < total_frames:
                labels[frame, string_num] = fret
    
    return labels


class GuitarSetDataset(Dataset):
    def __init__(self, data_root, context_window=9, split='all', audio_subdir=None):
        self.data_root = Path(data_root)
        
        audio_base = self.data_root / 'audio'
        
        if audio_subdir:
            self.audio_dir = audio_base / audio_subdir
        else:
            subdirs = [d for d in audio_base.iterdir() if d.is_dir()]
            if subdirs:
                self.audio_dir = subdirs[0]
            else:
                self.audio_dir = audio_base
        
        annotation_base = self.data_root / 'annotation'
        annotation_subdirs = [d for d in annotation_base.iterdir() if d.is_dir()]
        if annotation_subdirs:
            self.annotation_dir = annotation_subdirs[0]
        else:
            self.annotation_dir = annotation_base
        
        self.cache_dir = self.data_root / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.context_window = context_window
        self.halfwin = context_window // 2
        
        self.sr = 22050
        self.hop_length = 512
        self.n_bins = 192
        self.bins_per_octave = 24
        
        self.file_list = self._load_file_list(split)
        self.frame_index = self._build_frame_index()
    
    def _load_file_list(self, split):
        all_files = sorted(self.audio_dir.glob('*.wav'))
        
        if split == 'all':
            return all_files
        elif split.startswith('player_'):
            player_id = split.split('_')[1]
            return [f for f in all_files if f.stem.startswith(player_id)]
        elif split.startswith('no_player_'):
            player_id = split.split('_')[2]
            return [f for f in all_files if not f.stem.startswith(player_id)]
        
        return all_files
        
    def _build_frame_index(self):
        index = []
        print(f"Building frame index and pre-caching labels for {len(self.file_list)} files...")
        
        for file_idx, audio_path in enumerate(self.file_list):
            spec = self._load_or_compute_spectrogram(audio_path)
            num_frames = len(spec)
            
            jams_name = audio_path.stem
            for suffix in ['_hex_cln', '_hex_original', '_hex', '_mix', '_mic', '_original', '_debleeded']:
                jams_name = jams_name.replace(suffix, '')
            jams_path = self.annotation_dir / f"{jams_name}.jams"
            
            self._load_or_compute_labels(jams_path, num_frames)
            
            for frame_idx in range(num_frames):
                index.append((file_idx, frame_idx))
            
            if (file_idx + 1) % 10 == 0:
                print(f"Processed {file_idx + 1}/{len(self.file_list)} files")
        
        return index
    
    def _atomic_save(self, data, cache_path):
        import uuid
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.parent / f".tmp_{uuid.uuid4().hex}.npy"
        np.save(temp_path, data)
        temp_path.replace(cache_path)
    
    def _load_or_compute_spectrogram(self, audio_path):
        cache_path = self.cache_dir / f"{audio_path.stem}_cqt.npy"
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Warning: Corrupted cache {cache_path}, regenerating... (Error: {e})")
                if cache_path.exists():
                    cache_path.unlink()
        
        spec = preprocess_audio_to_cqt(audio_path, sr=self.sr, hop_length=self.hop_length,
                                       n_bins=self.n_bins, bins_per_octave=self.bins_per_octave)
        
        self._atomic_save(spec, cache_path)
        return spec
    
    def _load_or_compute_labels(self, jams_path, num_frames):
        cache_path = self.cache_dir / f"{jams_path.stem}_labels.npy"
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Warning: Corrupted cache {cache_path}, regenerating... (Error: {e})")
                if cache_path.exists():
                    cache_path.unlink()
        
        notes = load_jams_annotation(jams_path)
        labels = generate_frame_labels(notes, num_frames, self.hop_length, self.sr)
        
        self._atomic_save(labels, cache_path)
        return labels
    
    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        file_idx, frame_idx = self.frame_index[idx]
        
        audio_path = self.file_list[file_idx]
        spec = self._load_or_compute_spectrogram(audio_path)
        
        jams_name = audio_path.stem
        for suffix in ['_hex_cln', '_hex_original', '_hex', '_mix', '_mic', '_original', '_debleeded']:
            jams_name = jams_name.replace(suffix, '')
        jams_path = self.annotation_dir / f"{jams_name}.jams"
        labels = self._load_or_compute_labels(jams_path, len(spec))
        
        padded_spec = np.pad(spec, [(self.halfwin, self.halfwin), (0, 0)], mode='constant')
        window = padded_spec[frame_idx:frame_idx + self.context_window]
        
        x = window.T[np.newaxis, :, :]
        y = labels[frame_idx]
        
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
