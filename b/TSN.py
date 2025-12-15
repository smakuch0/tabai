import torch
import torch.nn as nn
import numpy as np


class MockTSN(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.randint(0, 25, (batch_size, 6), dtype=torch.float32)


class TSN:
    def __init__(self):
        self.num_frets = 24
        self.num_strings = 6
        self.num_classes = 25
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        self.model = MockTSN().to(self.device)
        self.model.eval()
    
    def load_weights(self, weights_path):
        pass
    
    def predict(self, audio_repr, context_window=9):
        if self.model is None:
            raise RuntimeError("Model not built")
        
        halfwin = context_window // 2
        predictions = []
        padded = np.pad(audio_repr, [(halfwin, halfwin), (0, 0)], mode='constant')
        
        with torch.no_grad():
            for i in range(len(audio_repr)):
                window = padded[i:i + context_window]
                x = np.expand_dims(np.expand_dims(np.swapaxes(window, 0, 1), -1), 0)
                x_tensor = torch.from_numpy(x).float().to(self.device)
                
                pred = self.model(x_tensor)
                predictions.append(pred.cpu().numpy()[0])
        
        return np.array(predictions, dtype=int)
