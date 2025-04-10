# stolen from https://github.com/neuroneural/meanMLP/blob/main/src/models/dice.py
""" DICE model from https://github.com/UsmanMahmood27/DICE """
from random import uniform, randint

import torch
from torch import nn
from torch import optim

from omegaconf import OmegaConf
from .utils import ModelOutputs


def default_HPs(args):
    model_cfg = {
        "lstm": {
            "bidirectional": True,
            "num_layers": 1,
            "hidden_size": 50,
        },
        "clf": {
            "hidden_size": 64,
            "num_layers": 0,
        },
        "MHAtt": {
            "n_heads": 1,
            "head_hidden_size": 48,
            "dropout": 0.0,
        },
        "scheduler": {
            "patience": 4,
            "factor": 0.5,
        },
        "reg_param": 1e-6,
        "lr": 2e-4,
        "input_size": args.num_roi,
        "output_size": 2,
    }
    return OmegaConf.create(model_cfg)


class DICE(nn.Module):
    """
    DICE model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(self, args):
        super().__init__()

        model_cfg = default_HPs(args)
        input_size = model_cfg.input_size
        output_size = model_cfg.output_size

        lstm_hidden_size = model_cfg.lstm.hidden_size
        lstm_num_layers = model_cfg.lstm.num_layers
        bidirectional = model_cfg.lstm.bidirectional

        self.lstm_output_size = (
            lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        )

        clf_hidden_size = model_cfg.clf.hidden_size
        clf_num_layers = model_cfg.clf.num_layers

        MHAtt_n_heads = model_cfg.MHAtt.n_heads
        MHAtt_hidden_size = MHAtt_n_heads * model_cfg.MHAtt.head_hidden_size
        MHAtt_dropout = model_cfg.MHAtt.dropout

        # LSTM - first block
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Classifier - last block
        clf = [
            nn.Linear(input_size**2, clf_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(clf_num_layers):
            clf.append(nn.Linear(clf_hidden_size, clf_hidden_size))
            clf.append(nn.ReLU())
        clf.append(
            nn.Linear(clf_hidden_size, output_size),
        )
        self.clf = nn.Sequential(*clf)

        # Multihead attention - second block
        self.key_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=MHAtt_hidden_size,
            num_heads=MHAtt_n_heads,
            dropout=MHAtt_dropout,
            batch_first=True,
        )

        # Global Temporal Attention - third block
        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()
        self.gta_embed = nn.Sequential(
            nn.Linear(
                input_size**2,
                round(self.upscale * input_size**2),
            ),
        )
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * input_size**2)),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * input_size**2),
                round(self.upscale2 * input_size**2),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * input_size**2), 1),
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.clf.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.query_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.key_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.value_layer.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.multihead_attn.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_embed.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")
        for name, param in self.gta_attend.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, mode="fan_in")

    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)

    def multi_head_attention(self, x):
        # x.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        key = self.key_layer(x)
        value = self.value_layer(x)
        query = self.query_layer(x)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)

        return attn_output, attn_output_weights

    def forward(self, data):
        x = data['timeseries']
        B, T, C = x.shape  # [batch_size, time_length, input_feature_size]

        # 1. pass input to LSTM; treat each channel as an independent single-feature time series
        x = x.permute(0, 2, 1)  # x.shape: [batch_size; input_feature_size; time_length]
        x = x.reshape(B * C, T, 1)  # x.shape: [batch_size * n_channels; time_length; 1]
        ##########################
        lstm_output, _ = self.lstm(x)
        # lstm_output.shape: [batch_size * input_feature_size; time_length; lstm_hidden_size]
        ##########################
        lstm_output = lstm_output.reshape(B, C, T, self.lstm_output_size)
        # lstm_output.shape: [batch_size; input_feature_size; time_length; lstm_hidden_size]

        # 2. pass lstm_output at each time point to multihead attention to reveal spatial connctions
        lstm_output = lstm_output.permute(2, 0, 1, 3)
        # lstm_output.shape: [time_length; batch_size; input_feature_size; lstm_hidden_size]
        lstm_output = lstm_output.reshape(T * B, C, self.lstm_output_size)
        # lstm_output.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        ##########################
        _, attn_weights = self.multi_head_attention(lstm_output)
        # attn_weights.shape: [time_length * batch_size; input_feature_size; input_feature_size]
        ##########################
        attn_weights = attn_weights.reshape(T, B, C, C)
        # attn_weights.shape: [time_length; batch_size; input_feature_size; input_feature_size]
        attn_weights = attn_weights.permute(1, 0, 2, 3)
        # attn_weights.shape: [batch_size; time_length; input_feature_size; input_feature_size]

        # 3. pass attention weights to a global temporal attention to obrain global graph
        attn_weights = attn_weights.reshape(B, T, -1)
        # attn_weights.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        ##########################
        FC = self.gta_attention(attn_weights)
        # FC.shape: [batch_size; input_feature_size * input_feature_size]
        ##########################

        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(FC)
        # logits.shape: [batch_size; n_classes]

        return ModelOutputs(logits=logits, features=FC)