import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Parameter
import math
import config as cfg
from torch.cuda import amp

# Implementation from https://github.com/HuangYG123/CurricularFace/blob/master/head/metrics.py

class ArcFace(nn.Module):
    def __init__(self,in_features,out_features, s=64.0, m=0.5, easy_margin=False):
        super(ArcFace,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.s = s
        self.m = m
        
        self.kernel = Parameter(torch.FloatTensor(in_features,out_features))
        # nn.init.normal_(self.kernel,std=0.01)
        nn.init.xavier_normal_(self.kernel)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)    
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        
    def forward(self,embeddings,label):
        if cfg.MODEL == 'Inception':
            # print(f'[INFO] Using {cfg.MODEL}. Hence no l2 norm is used since the embeddings are already normalized')
            #the pretrained Inception Model when classify=False returns the normalized feature embedddings. 
            embeddings = embeddings         
        else:
            embeddings = l2_norm(embeddings,axis=1)
            
        kernel_norm = l2_norm(self.kernel,axis=0)
        cos_theta = torch.mm(embeddings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1)
        
        with torch.no_grad():
            origin_cos = cos_theta.clone()
            
        target_logit = cos_theta[torch.arange(0,embeddings.size(0)),label].view(-1,1)
        
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit,2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        
        if self.easy_margin:
            final_target_logit = torch.where(target_logit>0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit> self.th, cos_theta_m, target_logit - self.mm)
            
                    
        cos_theta.scatter_(1,label.view(-1,1).long(), final_target_logit)
        
        output = cos_theta * self.s
            
        return output, origin_cos * self.s
    
            
def l2_norm(input,axis=1):
    #Equivalent to F.normalize(input,p=2)
    
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input,norm) 
    
    return output    
    


class CirriculumFace(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(CirriculumFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s