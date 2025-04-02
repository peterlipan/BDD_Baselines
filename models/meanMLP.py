# Stolen from https://github.com/neuroneural/meanMLP
import torch
from torch import nn
from .utils import ModelOutputs


class meanMLP(nn.Module):
    """
    meanMLP model for fMRI data.
    Expected input shape: [batch_size, time_length, n_components].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
    Data info expected in model_cfg:
        input_size: int - input n_components
        output_size: int - n_classes
    """

    def __init__(self, args):
        super().__init__()

        output_size = 2 # default for DX tasks
        dropout = 0.49
        hidden_size = 160

        self.hidden_size = hidden_size
        self.input_size = args.num_roi if args.fusion != 'early' else args.num_roi + args.num_phe

        layers = [
            nn.Linear(self.input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        # bs, tl, fs = x.shape  # [batch_size, time_length, input n_components]

        x = data['timeseries']

        features = self.encoder(x)
        features = features.mean(1)
        logits = self.classifier(features)

        return ModelOutputs(features=features, logits=logits)
