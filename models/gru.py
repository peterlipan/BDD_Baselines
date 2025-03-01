import torch
import torch.nn as nn
from .utils import ModelOutputs

class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_layers = 2
        self.hidden_size = 128
        self.gru = torch.nn.GRU(args.num_roi, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, 2)
    
    def forward(self, data):
        x = data['timeseries']
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        features = out[:, -1, :]
        logits = self.fc(features)
        return ModelOutputs(logits=logits, features=features)
