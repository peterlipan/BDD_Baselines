import torch
import torch.nn as nn


class ModelOutputs:
    def __init__(self, features=None, logits=None, **kwargs):
        self.dict = {'features': features, 'logits': logits}
        self.dict.update(kwargs)
    
    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
    
    def __str__(self):
        return str(self.dict)

    def __repr__(self):
        return str(self.dict)

    def __getattr__(self, key):
        return self.dict[key]


def get_model(args):
    if args.model.lower() == 'brainnetcnn':
        from .brainnetcnn import BrainNetCNN
        return BrainNetCNN(args)
    elif args.model.lower() == 'graphtransformer':
        from .GraphTransformer import GraphTransformer
        return GraphTransformer(args)
    elif args.model.lower() == 'bnt':
        from .bnt import BrainNetworkTransformer
        return BrainNetworkTransformer(args)
    elif args.model.lower() == 'fbnetgen':
        from .fbnetgen import FBNETGEN
        return FBNETGEN(args)
    elif args.model.lower() == 'comtf':
        from .ComTF import ComBrainTF
        return ComBrainTF(args)
    elif args.model.lower() == 'braingnn':
        from .braingnn import BrainGNN
        return BrainGNN(args)
    elif args.model.lower() == 'braingb':
        from .braingb import BrainGB
        return BrainGB(args)
    elif args.model.lower() == 'dpm':
        from .dual_pathway import mixed_model
        return mixed_model(args)
    elif args.model.lower() == 'lstm':
        from .lstm import LSTM
        return LSTM(args)
    elif args.model.lower() == 'transformer':
        from .ValinaTF import Transformer
        return Transformer(args)
    elif args.model.lower() == 'gru':
        from .gru import GRU
        return GRU(args)
    elif args.model.lower() == 'meanmlp':
        from .meanMLP import meanMLP
        return meanMLP(args)
    elif args.model.lower() == 'bolt':
        from .bolT import BolT
        return BolT(args)
    elif args.model.lower() == 'dice':
        from .DICE import DICE
        return DICE(args)
    elif args.model.lower() == 'glacier':
        from .glacier import Glacier
        return Glacier(args)
    else:
        raise NotImplementedError
    

class MultiModalFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        fusion_type = args.fusion
        assert fusion_type in ['early', 'intermediate', 'late', 'none', 'dpl'], 'Invalid fusion type'
        
        self.image_model = get_model(args)
        self.classifier = nn.Linear(self.image_model.hidden_size + args.num_phe, 2) if fusion_type == 'intermediate' else None
        self.fusion_type = fusion_type

    def forward(self, data):
        if self.fusion_type == 'early':
            ts = data['timeseries'] # B,T,N
            phi = data['phenotypes'] # B,P
            # repeat phi to match the time dimension
            phi = phi.unsqueeze(1).expand(-1, ts.size(1), -1)
            data['timeseries'] = torch.cat((ts, phi), dim=-1)
            return self.image_model(data)
        elif self.fusion_type == 'intermediate':
            outputs = self.image_model(data)
            features = outputs.features # B,C
            phi = data['phenotypes']
            features = torch.cat((features, phi), dim=-1)
            logits = self.classifier(features)
            return ModelOutputs(features=features, logits=logits)
        elif self.fusion_type == 'late':
            outputs = self.image_model(data)
            logits = outputs.logits # B,2
            features = outputs.features
            phi = data['phenotypes'].clone() # B,P
            phi = torch.mean(phi, dim=1, keepdim=True) # B,1
            logits = logits + phi
            return ModelOutputs(features=features, logits=logits)
        else:
            return self.image_model(data)