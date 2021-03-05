
import torch
import torch.nn as nn
import torch.nn.functional as F 
import config as cfg
from torch.autograd import Variable
import numpy as np 


class CBFocalLoss(nn.Module):
    # from https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    # paper https://arxiv.org/abs/1901.05555

    def __init__(self,weights,gamma):
        super(CBFocalLoss,self).__init__()
        self.alpha = weights     #weights per class for Class Balanced Loss or alpha=1 bare FocalLoss
        self.gamma = gamma
        
    def forward(self,logits,labels):
        labels = F.one_hot(labels,cfg.CLASSES).float()
        BCLoss = F.binary_cross_entropy_with_logits(logits,labels,weight=None,reduction='none')
            
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1+torch.exp(-1.0 * logits)))
        loss = modulator * BCLoss
        
        weighted_loss = self.alpha * loss
        focal_loss = torch.sum(weighted_loss)
        
        focal_loss /= torch.sum(labels)
        
        return focal_loss
            
class FocalLoss(nn.Module):
    def __init__(self,alpha= 1,gamma= 1,reduce= True,cls_weights=None):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.cls_weights = cls_weights
        
    def forward(self,logits,labels):
        CE_loss = F.cross_entropy(logits,labels,weight=self.cls_weights,reduction = "none")
        
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def LossFunctions(loss_fn_name,alpha,gamma,cls_weights=None):
    #Gamma value chosen from the paper https://arxiv.org/abs/1901.05555
    if not loss_fn_name:
        print(f'[INFO] Using default loss function - CrossEntropy Loss')
        loss_fn_name = 'ce'
        
    if loss_fn_name == 'cb_focal':
        loss = CBFocalLoss(weights = cls_weights, gamma = gamma)
        return loss
    
    if loss_fn_name == 'focal':
        print(f'[INFO] Using {loss_fn_name} as loss..')
        loss = FocalLoss(alpha,gamma,True,cls_weights)
        return loss

    if loss_fn_name == 'ce':
        loss = torch.nn.CrossEntropyLoss(cls_weights)
        return loss
    



# ------------------------------------------------------------------------------------------------------
# class CustomSigmoid(nn.Module):
#     def __init__(self,weights=None):
#         super(CustomSigmoid,self).__init__()
#         self.cls_weights = weights
    
#     def forward(self,logits,labels):
#         labels = F.one_hot(labels,cfg.CLASSES).float()
#         loss = F.binary_cross_entropy_with_logits(input=logits,target=labels,weight=self.cls_weights)
#         loss = torch.mean(loss)
#         return loss       
# ----------------------------------------------------------------------------------------------------      
    
# # testing
# torch.manual_seed(0)
# batch_size, n_classes = 8, 31
# x = torch.randn(batch_size, n_classes)
# batch_target = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)
# fn = ['focal']
# for fn_name in fn:
#     loss_fn = LossFunctions(fn_name)
#     print(f'[INFO] Function - {fn_name}, Loss - {loss_fn(x,batch_target)}')

# no_of_classes = 5
# logits = torch.rand(10,no_of_classes).float()
# labels = torch.randint(0,no_of_classes, size = (10,))
# beta = 0.9999
# gamma = 2.0
# samples_per_cls = [2,3,1,2,2]
# labels_one_hot = F.one_hot(labels, no_of_classes).float()
# effective_num = 1.0 - np.power(beta, samples_per_cls)
# weights = (1.0 - beta) / np.array(effective_num)
# weights = weights / np.sum(weights) * no_of_classes
# weights = torch.tensor(weights).float()
# weights = weights.unsqueeze(0)
# weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
# weights = weights.sum(1)
# weights = weights.unsqueeze(1)
# weights = weights.repeat(1,no_of_classes)
# print(weights.shape)
# print(logits.shape)
# print(labels_one_hot.shape)

# cb_loss = CBFocalLoss(weights = weights, gamma = gamma)
# loss = cb_loss(logits,labels_one_hot)
# print(loss)