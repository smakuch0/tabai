import torch
from b.data.dataset import GuitarSetDataset
from b.TSN import RealTSN

print("=== Integration Test: Training Setup ===\n")

print("1. Loading dataset...")
try:
    dataset = GuitarSetDataset(
        'b/data/GuitarSet',
        split='all',
        audio_subdir='audio_mono-pickup_mix'
    )
    print(f"     Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n2. Testing single sample...")
try:
    x, y = dataset[0]
    print(f"     Input shape: {x.shape}")
    print(f"     Label shape: {y.shape}")
    assert x.shape == (1, 192, 9), f"Wrong input shape: {x.shape}"
    assert y.shape == (6,), f"Wrong label shape: {y.shape}"
    print(f"     Shapes correct")
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n3. Testing batch creation...")
try:
    batch_x = torch.stack([dataset[i][0] for i in range(4)])
    batch_y = torch.stack([dataset[i][1] for i in range(4)])
    print(f"     Batch X shape: {batch_x.shape}")
    print(f"     Batch Y shape: {batch_y.shape}")
    assert batch_x.shape == (4, 1, 192, 9)
    assert batch_y.shape == (4, 6)
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n4. Testing model forward pass...")
try:
    model = RealTSN()
    model.eval()
    with torch.no_grad():
        output = model(batch_x)
    print(f"     Output shape: {output.shape}")
    assert output.shape == (4, 6, 25), f"Wrong output shape: {output.shape}"
    print(f"     Forward pass successful")
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n5. Testing loss computation...")
try:
    from b.train import compute_loss, compute_accuracy
    loss = compute_loss(output, batch_y)
    acc = compute_accuracy(output, batch_y)
    print(f"     Loss: {loss:.4f}")
    print(f"     Accuracy: {acc:.4f}")
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n6. Testing splits...")
try:
    train_ds = GuitarSetDataset('b/data/GuitarSet', split='no_player_00', audio_subdir='audio_mono-pickup_mix')
    val_ds = GuitarSetDataset('b/data/GuitarSet', split='player_00', audio_subdir='audio_mono-pickup_mix')
    print(f"     Train: {len(train_ds)} samples")
    print(f"     Val: {len(val_ds)} samples")
    print(f"     Ratio: {len(val_ds)/len(dataset)*100:.1f}% validation")
except Exception as e:
    print(f"      Error: {e}")
    exit(1)

print("\n" + "="*50)
print("  All tests passed! Ready for training.")
print("="*50)
print("\nRun: python -m b.train --epochs 10")
