import torch
import numpy as np
import random
from pathlib import Path
from b.data.dataset import GuitarSetDataset

print("Testing GuitarSetDataset...")

data_root = Path(__file__).parent.parent / 'data' / 'GuitarSet'

dataset = GuitarSetDataset(
    data_root=data_root,
    context_window=9,
    split='all',
    audio_subdir='audio_hex-pickup_debleeded'
)

print(f"Dataset size: {len(dataset)}")
print(f"Audio dir: {dataset.audio_dir}")
print(f"Annotation dir: {dataset.annotation_dir}")
print(f"Files found: {len(dataset.file_list)}")

if len(dataset) > 0:
    x, y = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Input shape: {x.shape}")
    print(f"  Label shape: {y.shape}")
    print(f"  Input dtype: {x.dtype}")
    print(f"  Label dtype: {y.dtype}")
    print(f"  Label range: [{y.min()}, {y.max()}]")
    
    assert x.shape[0] == 1, f"Wrong channel dimension: {x.shape[0]}"
    assert x.shape[1] == 192, f"Wrong frequency bins: {x.shape[1]}"
    assert x.shape[2] == 9, f"Wrong context window: {x.shape[2]}"
    assert y.shape[0] == 6, f"Wrong number of strings: {y.shape[0]}"
    assert y.min() >= 0 and y.max() <= 24, "Labels out of range"
    
    print("\nTesting batch loading...")
    batch_size = 8
    batch_indices = list(range(min(batch_size, len(dataset))))
    
    xs = []
    ys = []
    for idx in batch_indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    
    x_batch = torch.stack(xs)
    y_batch = torch.stack(ys)
    
    print(f"  Batch input shape: {x_batch.shape}")
    print(f"  Batch label shape: {y_batch.shape}")
    
    assert x_batch.shape == (len(batch_indices), 1, 192, 9)
    assert y_batch.shape == (len(batch_indices), 6)
    
    print("\nTesting split modes...")
    if len(dataset.file_list) > 0:
        first_file = dataset.file_list[0]
        player_id = first_file.stem[:2]
        
        player_dataset = GuitarSetDataset(
            data_root=data_root,
            context_window=9,
            split=f'player_{player_id}',
            audio_subdir='audio_hex-pickup_debleeded'
        )
        print(f"  Player {player_id} dataset size: {len(player_dataset)}")
        
        no_player_dataset = GuitarSetDataset(
            data_root=data_root,
            context_window=9,
            split=f'no_player_{player_id}',
            audio_subdir='audio_hex-pickup_debleeded'
        )
        print(f"  Without player {player_id} dataset size: {len(no_player_dataset)}")

        for _ in range(10):
            idx = random.randint(0, len(dataset)-1)
            x, y = dataset[idx]
            print(f"Sample {idx}: label range [{y.min()}, {y.max()}], values: {y.numpy()}")
