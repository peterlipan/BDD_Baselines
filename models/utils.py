import torch


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
    else:
        raise NotImplementedError
    