import torch
import torch.nn as nn
from .utils import ModelOutputs

class GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_layers = 2
        self.hidden_size = 256
        self.input_size = args.num_roi if args.fusion != 'early' else args.num_roi + args.num_phe
        self.gru = torch.nn.GRU(self.input_size, 128, self.n_layers, 
                                batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(self.hidden_size, 2)
    
    def forward(self, data):
        x = data['timeseries']
        h0 = torch.zeros(self.n_layers, x.size(0), 128).to(x.device)
        
        out, _ = self.gru(x, h0)
        features = out[:, -1, :]
        logits = self.fc(features)
        return ModelOutputs(logits=logits, features=features)
