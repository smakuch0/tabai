import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa


class RealTSN(nn.Module):
    def __init__(self, context_window=9):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.fc_input_size = 128 * 24 * 1
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout = nn.Dropout(0.5)
        
        self.string_outputs = nn.ModuleList([
            nn.Linear(512, 25) for _ in range(6)
        ])
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        outputs = []
        for string_classifier in self.string_outputs:
            outputs.append(string_classifier(x))
        
        return torch.stack(outputs, dim=1)


class TSN:
    def __init__(self):
        self.num_frets = 24
        self.num_strings = 6
        self.num_classes = 25
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 22050
        self.hop_length = 512
        self.n_bins = 192
        self.bins_per_octave = 24
    
    def build_model(self):
        self.model = RealTSN(context_window=9).to(self.device)
        self.model.eval()
    
    def load_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def preprocess_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        y = librosa.util.normalize(y)
        
        spec = np.abs(librosa.cqt(
            y, 
            sr=sr, 
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        ))
        
        spec = np.log(spec + 1e-10)
        return np.swapaxes(spec, 0, 1)
    
    def predict(self, audio_repr, context_window=9):
        if self.model is None:
            raise RuntimeError("Model not built")
        
        halfwin = context_window // 2
        predictions = []
        padded = np.pad(audio_repr, [(halfwin, halfwin), (0, 0)], mode='constant')
        
        with torch.no_grad():
            for i in range(len(audio_repr)):
                window = padded[i:i + context_window]
                x = window.T
                x = x[np.newaxis, np.newaxis, :, :]
                x_tensor = torch.from_numpy(x).float().to(self.device)
                
                logits = self.model(x_tensor)
                pred = torch.argmax(logits, dim=-1)
                predictions.append(pred.cpu().numpy()[0])
        
        return np.array(predictions, dtype=int)
