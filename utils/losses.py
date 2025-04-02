import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, scale_factors=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # normalize the features
        features = F.normalize(features, dim=-1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]            
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # scale the logits (cosine sim) by the scale factors (cross-modal sim)
        if scale_factors is not None:
            scale_factors = scale_factors.repeat(anchor_count, contrast_count)
            logits = logits * scale_factors

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class TaskOrientedPhenotypeLearning(nn.Module):
    def __init__(self, world_size, batch_size, temperature=0.1, alpha=5.0):
        super().__init__()
        self.world_size = world_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrastive_loss = SupConLoss(temperature)
        self.alpha = alpha

    def compute_categorical_sim(self, cat_phenotypes):
        device = cat_phenotypes.device
        batch_size, n_features = cat_phenotypes.shape
        
        similarity = torch.ones(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cat_phenotypes[:, i]            
            valid_mask = (feature != -1)
            confidence = valid_mask.float().sum() / batch_size # confidence based on the ratio of valid values
            
            if valid_mask.sum() <= 1:
                # Skip this feature if there are not enough valid values
                continue
            
            # Compute similarity for this feature
            feature_sim = torch.zeros(batch_size, batch_size, device=device)
            
            # Get valid indices
            valid_indices = torch.where(valid_mask)[0]
            
            # Compute similarity only for valid pairs
            for idx1, i in enumerate(valid_indices):
                for idx2, j in enumerate(valid_indices):
                    if feature[i] == feature[j]:
                        feature_sim[i, j] = 1.0 * confidence
            
            # Update overall similarity
            similarity += feature_sim
        
        return similarity / n_features

    def compute_continuous_sim(self, cont_phenotypes):
        """
        Compute similarity for continuous phenotypes with careful handling of missing values
        """
        device = cont_phenotypes.device
        batch_size, n_features = cont_phenotypes.shape
        
        # Initialize similarity matrix
        similarity = torch.ones(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cont_phenotypes[:, i]            
            valid_mask = (feature != -1)
            confidence = valid_mask.float().sum() / batch_size
            
            if valid_mask.sum() <= 1:
                continue
                
            valid_values = feature[valid_mask]
            
            mean = valid_values.mean()
            std = valid_values.std() + 1e-6
            
            # Normalize valid values
            normalized = (valid_values - mean) / std
            
            # Compute pairwise distances for valid values
            dist_matrix = torch.zeros(batch_size, batch_size, device=device)
            valid_indices = torch.where(valid_mask)[0]
            
            for idx1, i in enumerate(valid_indices):
                for idx2, j in enumerate(valid_indices):
                    if i != j:
                        dist = torch.square(normalized[idx1] - normalized[idx2])
                        dist_matrix[i, j] = dist
            
            # Convert distances to similarities using Gaussian kernel
            sigma = 1.0
            sim = torch.exp(-dist_matrix / (2 * sigma * sigma)) * confidence
            
            # Update overall similarity
            similarity += sim
        
        return similarity / n_features

    def compute_scale_factors(self, cat_phenotypes=None, cont_phenotypes=None):
        if cat_phenotypes is None and cont_phenotypes is None:
            return None
        bs = cat_phenotypes.shape[0] if cat_phenotypes is not None else cont_phenotypes.shape[0]
        device = cat_phenotypes.device if cat_phenotypes is not None else cont_phenotypes.device

        phenotype_sim = torch.ones((bs, bs), device=device) # base scale_factor = 1
        
        if cat_phenotypes is not None:
            cat_sim = self.compute_categorical_sim(cat_phenotypes)
            cat_sim[cat_sim > 1] *= self.alpha
            cat_sim[cat_sim < 1] /= self.alpha
            phenotype_sim *= cat_sim
            
        if cont_phenotypes is not None:
            cont_sim = self.compute_continuous_sim(cont_phenotypes)
            cont_sim[cont_sim > 1] *= self.alpha
            cont_sim[cont_sim < 1] /= self.alpha
            phenotype_sim *= cont_sim    


        return phenotype_sim

    def forward(self, features, labels=None, cat_phenotypes=None, cont_phenotypes=None):

        
        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0) if labels is not None else None
            cat_phenotypes = torch.cat(GatherLayer.apply(cat_phenotypes), dim=0) if cat_phenotypes is not None else None
            cont_phenotypes = torch.cat(GatherLayer.apply(cont_phenotypes), dim=0) if cont_phenotypes is not None else None
        
        scale_factors = self.compute_scale_factors(cat_phenotypes, cont_phenotypes)
        
        loss = self.contrastive_loss(features, labels, scale_factors)
        if labels is not None:
            loss += self.contrastive_loss(features, None, scale_factors)

        return loss


class bolTLoss:
    """Cross-entropy loss with model regularization"""

    def __init__(self):
        self.ce_loss = nn.CrossEntropyLoss()

        self.lambdaCons = 1 # default value

    def __call__(self, logits, target, cls):
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        cross_entropy_loss = self.ce_loss(logits, target)

        return cross_entropy_loss + clsLoss * self.lambdaCons


def mixup_cluster_loss(matrixs, y, intra_weight=2):

    y_1 = y[:, 1].float()

    y_0 = y[:, 0].float()

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+1e-10).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+1e-10).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.cuda()
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res


def braingnn_loss(output, w1, w2, s1, s2, label):
        # print(f'output.dtype={output.dtype},w1.dtype={w1.dtype}, w2.dtype={w2.dtype}, s1.dtype={s1.dtype}, s2.dtype={s2.dtype}, label.dtype={label.dtype}')
        s1, s2 = s1.float(), s2.float()


        major_loss = torch.nn.CrossEntropyLoss()(output,label)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1, 0.5)
        loss_tpk2 = topk_loss(s2, 0.5)
        loss_consist = 0


        for c in range(2):
            loss_consist += consist_loss(s1[label == c])

        loss = major_loss + 0.1 * loss_tpk1 + 0.1 * loss_tpk2 + 0.1 * loss_consist
        
        return loss


class LossFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = args.model
        self.bolt_loss = bolTLoss() if self.model.lower() == 'bolt' else None
        self.topl = TaskOrientedPhenotypeLearning(args.world_size, args.batch_size)
        self.fusion = args.fusion
    
    def forward(self, outputs, data):
        if self.model.lower() == 'fbnetgen':
            loss = self.ce(outputs.logits, data['label'])
            loss += mixup_cluster_loss(outputs.learnable_matrix, data['onehot'])
            loss += 1.0e-4 * torch.norm(outputs.learnable_matrix, p=1)
        elif self.model.lower() == 'braingnn':
            loss = braingnn_loss(outputs.logits, outputs.w1, outputs.w2, outputs.s1, outputs.s2, data['label'])
        elif self.model.lower() == 'bolt':
            loss = self.bolt_loss(outputs.logits, data['label'], outputs.cls)
        else:
            loss = self.ce(outputs.logits, data['label'])
        
        if self.fusion == 'dpl':
            loss += 0.03 * self.topl(outputs.features.unsqueeze(1), labels=data['label'], 
                                     cat_phenotypes=data['cp'], cont_phenotypes=data['cnp'])
        
        return loss
