import torch
from .utils import ModelOutputs

class LSTM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_layers = 2
        self.hidden_size = 128
        self.lstm = torch.nn.LSTM(args.num_roi, self.hidden_size, self.n_layers, batch_first=True)
        self.fc = torch.nn.Linear(128, 2)

    def forward(self, data):
        x = data['timeseries']
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        features = out[:, -1, :]
        logits = self.fc(features)
        return ModelOutputs(logits=logits, features=features)
