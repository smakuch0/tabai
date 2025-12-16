import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from b.data.dataset import GuitarSetDataset
from b.TSN import RealTSN


def compute_loss(logits, targets):
    criterion = nn.CrossEntropyLoss()
    loss = 0
    for string_idx in range(6):
        loss += criterion(logits[:, string_idx, :], targets[:, string_idx])
    return loss / 6


def compute_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=2)
    correct = (predictions == targets).float()
    return correct.mean().item()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    for batch_idx, (cqt, target) in enumerate(tqdm(dataloader, desc="Training")):
        cqt = cqt.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        logits = model(cqt)
        loss = compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += compute_accuracy(logits, target)
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for cqt, target in tqdm(dataloader, desc="Validating"):
            cqt = cqt.to(device)
            target = target.to(device)
            
            logits = model(cqt)
            loss = compute_loss(logits, target)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(logits, target)
    
    return total_loss / len(dataloader), total_acc / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='b/data/GuitarSet')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir', type=str, default='b/checkpoints')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("Loading datasets...")
    train_dataset = GuitarSetDataset(
        args.data, 
        split='no_player_00',
        audio_subdir='audio_mono-pickup_mix'
    )
    val_dataset = GuitarSetDataset(
        args.data, 
        split='player_00',
        audio_subdir='audio_mono-pickup_mix'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    model = RealTSN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': history
        }
        
        torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch}.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"Saved best model (acc: {val_acc:.4f})")
    
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
