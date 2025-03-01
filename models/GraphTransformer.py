import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .utils import ModelOutputs


class GraphTransformer(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.attention_list = nn.ModuleList()
        self.readout = 'concat'
        self.node_num = args.num_roi

        for _ in range(2):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=args.num_roi, nhead=3, dim_feedforward=1024,
                                        batch_first=True)
            )

        final_dim = args.num_roi

        if self.readout == "concat":
            self.dim_reduction = nn.Sequential(
                nn.Linear(args.num_roi, 8),
                nn.LeakyReLU()
            )
            final_dim = 8 * self.node_num

        elif self.readout == "sum":
            self.norm = nn.BatchNorm1d(args.num_roi)

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, data):
        node_feature = data['corr']
        bz, _, _, = node_feature.shape

        for atten in self.attention_list:
            node_feature = atten(node_feature)

        if self.readout == "concat":
            node_feature = self.dim_reduction(node_feature)
            node_feature = node_feature.reshape((bz, -1))

        elif self.readout == "mean":
            node_feature = torch.mean(node_feature, dim=1)
        elif self.readout == "max":
            node_feature, _ = torch.max(node_feature, dim=1)
        elif self.readout == "sum":
            node_feature = torch.sum(node_feature, dim=1)
            node_feature = self.norm(node_feature)
        
        logits = self.fc(node_feature)

        return ModelOutputs(features=node_feature, logits=logits)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]