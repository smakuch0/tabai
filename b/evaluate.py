import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm

from b.data.dataset import GuitarSetDataset
from b.TSN import RealTSN
from b.utils import midi_to_fret


def fret_string_accuracy(predictions, targets):
    correct = (predictions == targets).all(dim=1).float()
    return correct.mean().item()


def pitch_accuracy(predictions, targets):
    pred_pitches = []
    target_pitches = []
    
    for i in range(predictions.shape[0]):
        for string_idx in range(6):
            pred_fret = predictions[i, string_idx].item()
            target_fret = targets[i, string_idx].item()
            
            if target_fret > 0:
                open_string_midi = [40, 45, 50, 55, 59, 64]
                pred_pitch = open_string_midi[string_idx] + pred_fret
                target_pitch = open_string_midi[string_idx] + target_fret
                
                pred_pitches.append(pred_pitch)
                target_pitches.append(target_pitch)
    
    if len(target_pitches) == 0:
        return 0.0
    
    correct = sum([1 for p, t in zip(pred_pitches, target_pitches) if p == t])
    return correct / len(target_pitches)


def per_string_metrics(predictions, targets):
    metrics = {}
    
    for string_idx in range(6):
        pred_string = predictions[:, string_idx]
        target_string = targets[:, string_idx]
        
        playing = target_string > 0
        
        tp = ((pred_string > 0) & (target_string > 0)).sum().item()
        fp = ((pred_string > 0) & (target_string == 0)).sum().item()
        fn = ((pred_string == 0) & (target_string > 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        string_names = ['E', 'A', 'D', 'G', 'B', 'e']
        metrics[f'string_{string_idx}_{string_names[string_idx]}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for cqt, target in tqdm(dataloader, desc="Evaluating"):
            cqt = cqt.to(device)
            target = target.to(device)
            
            logits = model(cqt)
            predictions = torch.argmax(logits, dim=2)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    fs_acc = fret_string_accuracy(predictions, targets)
    pitch_acc = pitch_accuracy(predictions, targets)
    tdr = pitch_acc / fs_acc if fs_acc > 0 else 0
    
    per_string = per_string_metrics(predictions, targets)
    
    return {
        'fret_string_accuracy': fs_acc,
        'pitch_accuracy': pitch_acc,
        'tdr': tdr,
        'per_string': per_string
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, default='b/data/GuitarSet')
    parser.add_argument('--split', type=str, default='player_00')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    dataset = GuitarSetDataset(
        args.data, 
        split=args.split,
        audio_subdir='audio_mono-pickup_mix'
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print("Loading model...")
    model = RealTSN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    
    results = evaluate(model, dataloader, device)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Fret-String Accuracy: {results['fret_string_accuracy']:.4f}")
    print(f"Pitch Accuracy: {results['pitch_accuracy']:.4f}")
    print(f"TDR: {results['tdr']:.4f}")
    print("\nPer-String Metrics:")
    
    for string_name, metrics in results['per_string'].items():
        print(f"  {string_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


if __name__ == '__main__':
    main()
