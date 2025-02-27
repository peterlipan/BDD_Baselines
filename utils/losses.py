import torch
import torch.nn as nn


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


class LossFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='sum')
        self.model = args.model
    
    def forward(self, outputs, data):
        if self.model.lower() == 'fbnetgen':
            loss = self.ce(outputs.logits, data['label'])
            loss += mixup_cluster_loss(outputs.learnable_matrix, data['onehot'])
            loss += 1.0e-4 * torch.norm(outputs.learnable_matrix, p=1)
        else:
            loss = self.ce(outputs.logits, data['label'])
        return loss
