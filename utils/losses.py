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
            loss_consist += consist_loss(s1[label[:,1] == c])

        loss = major_loss + 0.1 * loss_tpk1 + 0.1 * loss_tpk2 + 0.1 * loss_consist
        
        return loss


class LossFunction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.model = args.model
    
    def forward(self, outputs, data):
        if self.model.lower() == 'fbnetgen':
            loss = self.ce(outputs.logits, data['label'])
            loss += mixup_cluster_loss(outputs.learnable_matrix, data['onehot'])
            loss += 1.0e-4 * torch.norm(outputs.learnable_matrix, p=1)
        elif self.model.lower() == 'braingnn':
            loss = braingnn_loss(outputs.logits, outputs.w1, outputs.w2, outputs.s1, outputs.s2, data['label'])
        else:
            loss = self.ce(outputs.logits, data['label'])
        return loss
