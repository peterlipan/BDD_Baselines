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
    else:
        raise NotImplementedError