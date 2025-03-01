import torch
import torch.nn as nn
from .utils import ModelOutputs


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_in = args.num_roi
        self.d_model = args.num_roi
        self.n_layers = 2
        self.embedding = nn.Linear(self.d_in, self.d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=args.n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.fc = nn.Linear(self.d_model, 2)

    def forward(self, data):
        x = data['timeseries']
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        features = x.mean(dim=1)
        logits = self.fc(features)
        return ModelOutputs(logits=logits, features=features)
