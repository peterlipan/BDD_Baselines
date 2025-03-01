import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from .metrics import compute_avg_metrics
import torch.distributed as dist
from .losses import LossFunction
from torch.utils.data import DataLoader
from datasets import AbideROIDataset, Transforms, AdhdROIDataset
from models import get_model
from sklearn.model_selection import KFold, StratifiedKFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup


class Trainer:
    def __init__(self, args, logger=None):

        self.logger = logger
        self.args = args
        self.result_csv_name = f"./results_{args.dataset}_{args.task}.csv"
    
    def init_adhd_datasets(self, args):
        train_csv = pd.read_csv(args.train_csv)
        test_csv = pd.read_csv(args.test_csv)
        self.train_dataset = AdhdROIDataset(train_csv, args.data_root, atlas=args.atlas,
                                            cp=args.cp, cnp=args.cnp, task=args.task)

        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            train_sampler = None

        args.num_cp = self.train_dataset.num_cp
        args.num_cnp = self.train_dataset.num_cnp

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                        drop_last=True, num_workers=args.workers, sampler=train_sampler, pin_memory=True,)
        
        n_clasees = self.train_dataset.n_classes
        self.n_classes = n_clasees
        args.n_classes = n_clasees

        if args.rank == 0:
            self.test_dataset = AdhdROIDataset(test_csv, args.data_root, atlas=args.atlas,
                                               cp=args.cp, cnp=args.cnp, task=args.task)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                         num_workers=args.workers, pin_memory=True)
            self.val_loader = None
        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_abide_datasets(self, args):
        train_csv = pd.read_csv(args.train_csv)
        test_csv = pd.read_csv(args.test_csv)
        val_csv = pd.read_csv(args.val_csv)
        self.train_dataset = AbideROIDataset(train_csv, args.data_root, atlas=args.atlas, task=args.task,
                                             cp=args.cp, cnp=args.cnp)

        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            train_sampler = None
        
        args.num_cp = self.train_dataset.num_cp
        args.num_cnp = self.train_dataset.num_cnp

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                       drop_last=True, num_workers=args.workers, sampler=train_sampler, pin_memory=True,)
        
        n_classes = self.train_dataset.n_classes
        self.n_classes = n_classes
        args.n_classes = n_classes
        
        if args.rank == 0:
            self.test_dataset = AbideROIDataset(test_csv, args.data_root, atlas=args.atlas, task=args.task,
                                                cp=args.cp, cnp=args.cnp)
            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True)
            
            self.val_dataset = AbideROIDataset(val_csv, args.data_root, atlas=args.atlas, task=args.task,
                                               cp=args.cp, cnp=args.cnp)
            self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                          num_workers=args.workers, pin_memory=True)
        else:
            self.test_loader = None
            self.val_loader = None
    
    def init_model(self, args):
        
        step_per_epoch = len(self.train_dataset) // (args.batch_size * args.world_size)
        self.model = get_model(args).cuda()

        self.optimizer = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr,
                                                              weight_decay=args.weight_decay)
        
        self.criterion = LossFunction(args).cuda()

        if args.scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup_epochs * step_per_epoch, 
                                                             args.epochs * step_per_epoch)
        else:
            self.scheduler = None
        
        if args.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[args.rank], static_graph=True)

    def validate(self, loader):
        self.model.eval()
        ground_truth = torch.Tensor().cuda()
        predictions = torch.Tensor().cuda()
        with torch.no_grad():
            for data in loader:
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}
                outputs = self.model(data)
                pred = F.softmax(outputs.logits, dim=-1)
                ground_truth = torch.cat((ground_truth, data['label']))
                predictions = torch.cat((predictions, pred))
            
            metric = compute_avg_metrics(ground_truth, predictions, avg='micro')
        self.model.train()
        return metric

    def train(self, args):
        cur_iter = 0
        for epoch in range(args.epochs):
            self.model.train()
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            for data in self.train_loader:
                data = {key: value.cuda(non_blocking=True) for key, value in data.items()}

                outputs = self.model(data) 
                loss = self.criterion(outputs, data)

                self.optimizer.zero_grad()
                loss.backward()

                if dist.is_available() and dist.is_initialized():
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                            p.grad.data /= dist.get_world_size()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
                cur_iter += 1
                if cur_iter % 10 == 1:
                    if args.rank == 0:
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        test_metrics = self.validate(self.test_loader)
                        val_metrics = self.validate(self.val_loader) if self.val_loader is not None else {'Accuracy': 0.0}
                        print(f'Epoch: {epoch}, Iter: {cur_iter}, LR: {cur_lr}, Acc: {test_metrics['Accuracy']}')
                        if self.logger is not None:
                            self.logger.log({'test': test_metrics, 'val': val_metrics,
                                             'train': {
                                                 'lr': cur_lr,
                                                'loss': loss.item(),}})
        if args.rank == 0:
            self.save_model(args)

    def run(self, args):
        if 'ABIDE' in args.dataset:
            self.init_abide_datasets(args)
        else:
            self.init_adhd_datasets(args)
        self.init_model(args)
        self.train(args)
        if args.rank == 0:
            metrics = self.validate(self.test_loader)
            print(f'Final: {metrics}')
            self.save_results(args, metrics)

    def save_model(self, args):
        state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        performance = self.validate(self.test_loader)
        save_path = os.path.join(args.checkpoints, f"{args.model}_{args.dataset}_{args.atlas}_{args.task}_AUC_{performance['AUC']:.4f}_.pth")
        torch.save(state_dict, save_path)
    
    def load_model(self, args):
        model_prefix = f"{args.dataset}_{args.atlas}_{args.task}"
        candidates = [f for f in os.listdir(args.checkpoints) if f.startswith(model_prefix)]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No pretrained model for condition {model_prefix}")
        candidates.sort()
        model_path = os.path.join(args.checkpoints, candidates[-1])
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def save_results(self, args, metrics):
        cols = ['Model', 'Dataset', 'Atlas', 'Task', 'Seed'] + list(metrics.keys())
        if not os.path.exists(self.result_csv_name):
            results = pd.DataFrame(columns=cols)
        else:
            results = pd.read_csv(self.result_csv_name)
            assert set(results.columns) == set(cols), "Columns mismatch"
        row = [args.model, args.dataset, args.atlas, args.task, args.seed] + [metrics[key] for key in metrics.keys()]
        results = results._append(pd.Series(row, index=cols), ignore_index=True)
        results.to_csv(self.result_csv_name, index=False)
