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

        input_size = args.num_roi
        output_size = 2 # default for DX tasks
        dropout = 0.49
        hidden_size = 160

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
        ]

        self.fc = nn.Sequential(*layers)

    def forward(self, data):
        # bs, tl, fs = x.shape  # [batch_size, time_length, input n_components]

        x = data['timeseries']

        fc_output = self.fc(x)
        logits = fc_output.mean(1)

        return ModelOutputs(features=fc_output, logits=logits)
