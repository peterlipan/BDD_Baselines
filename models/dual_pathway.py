import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from types import SimpleNamespace
from .components import InterpretableTransformerEncoder, Exp_InterpretableTransformerEncoder, gatv2_Exp_InterpretableTransformerEncoder, MyGNN
import torch.nn.functional as F
import math
from .Learnable_Nodefeature import Centrality_Nodefeature, SAN_Nodefeature, GCN_Nodefeature, DegreeBin_Nodefeature, TimeSeriesEncoder
from .utils import ModelOutputs
from .bnt import TransPoolingEncoder

class Exp_TransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True, num_heads=4):
        super().__init__()
        self.transformer = Exp_InterpretableTransformerEncoder(d_model=input_feature_size, nhead=num_heads,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)


        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)


    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x, edge_mask, mask_pos, is_explain, attn_map_bias, attn_solid_mask):

        x = self.transformer(x, attn_map_bias = attn_map_bias, attn_solid_mask = attn_solid_mask)        
    
        if self.pooling:
            x, assignment = self.dec(x)

            return x, assignment

        
        return x, None


    def get_attention_weights(self):
        return self.transformer.get_attention_weights()


    def loss(self, assignment):
        return self.dec.loss(assignment)


class mixed_model(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        self.attention_list = nn.ModuleList()

        # Using default settings
        self.forward_dim = args.num_roi
        self.has_self_loop = False
        self.has_nonaggr_module = True
        self.has_aggr_module = True
        self.aggr_combine_type = 'concat'
        self.gnn_hidden_channels = 128
        self.aggr_module = 'gat'
        self.gnn_pool = 'concat'
        self.one_layer_fc = True
        self.node_feature_type = 'learnable_time_series'
        self.node_feature_dim = 100
        self.num_bins = 100
        self.time_series_hidden_size = 128
        self.time_series_encoder = 'cnn'
        self.timeseries_embedding_type = 'last'
        self.node_feature_eigen_topk = 10
        self.time_series_input_size = 100
        self.gnn_num_layers = 1
        self.gnn_num_heads = 2
        self.nonaggr_type = 'input'

        ############  only used for BNT model, not for others  #######
        sizes = [360, 100] 
        in_sizes = [args.num_roi] + sizes[:-1]
        do_pooling = [False, True]

        self.do_pooling = do_pooling
        
        last_output_dim = 2

        num_nodes = args.num_roi

        residual_input_dim = int((num_nodes*(num_nodes-1))/2) # do not take diagonal into consideration
        

        fc_input_dim = residual_input_dim + self.gnn_hidden_channels * num_nodes                    
        

        self.nonaggr_fc = nn.Linear(residual_input_dim, last_output_dim, bias=True)
        
        self.aggr_fc = nn.Linear(self.gnn_hidden_channels*num_nodes, last_output_dim, bias=True)


        # learnable node feature

        self.node_feature_generator = TimeSeriesEncoder(input_size=1, hidden_size=self.time_series_hidden_size, 
                                                        embedding_size=self.node_feature_dim, 
                                                        model_type=self.time_series_encoder, 
                                                        timeseries_embedding_type=self.timeseries_embedding_type)

        

        gnn_inchannels = self.node_feature_dim

        self.gnn_module = MyGNN(inchannels=gnn_inchannels, hidden_channels=self.gnn_hidden_channels, 
                                num_layers=self.gnn_num_layers, gnn_type=self.aggr_module, 
                                num_heads=self.gnn_num_heads, eps=0.0, aggr='mean',)

        self.norm0 = nn.LayerNorm(self.gnn_hidden_channels)

        self.nonaggr_coefficient = 1
        self.aggr_coefficient = 1


    def forward(self,
                data,
                edge_mask=None,
                mask_pos=None,
                is_explain=False,
                ):
        
        
        timeseries = data['timeseries'].permute(0, 2, 1) # [B, T, N] -> [B, N, T]
        sparse_connection = data['sparse_connection']
        orig_connection = data['corr']
        bz, _, _, = timeseries.shape

        assignments = []

        # learnable node feature from time series
        node_feature = self.node_feature_generator(timeseries)

        # obtain attn_solid_mask from sparse_connection
        attn_solid_mask = torch.full_like(sparse_connection, float('-inf')).cuda()
        attn_solid_mask[sparse_connection != 0] = 0


        layer_cnt = 0
        LM_feature = orig_connection

        aggr_feature = self.gnn_module(node_feature, sparse_connection)

        aggr_feature = self.norm0(aggr_feature)
        

        aggr_feature = aggr_feature.reshape((bz, -1))
             
        num_nodes = orig_connection.shape[-1]
        offset = 1
        rows, cols = torch.triu_indices(row=num_nodes, col=num_nodes, offset=offset)  # offset 0 including diag

        LM_feature = LM_feature[:, rows, cols]

        aggr_feature = aggr_feature.reshape((bz, -1))
                        
        # separate fc into two parts
        nonaggr_output = self.nonaggr_fc(self.nonaggr_coefficient * LM_feature)
        aggr_output = self.aggr_fc(self.aggr_coefficient * aggr_feature)
        logits = nonaggr_output + aggr_output

    
        return ModelOutputs(logits=logits)


    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]
    

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all