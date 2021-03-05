import os
import pickle
import torch


def get_cls_to_idx(datasetpath):
    i=0
    map_dict = {}
    for foldernames in os.listdir(datasetpath):
        map_dict[foldernames] = i
        i+=1
    return map_dict

def get_idx_to_cls(cls_to_idx_dict):
    cls_to_idx = {v:k for (k,v) in cls_to_idx_dict.items()}
    
    return cls_to_idx

def create_dict(X,y):
    assert len(X) == len(y)
    
    xy_dict = {}
    for i in range(len(X)):
        emp_name = X[i].split('\\')[-2]
        xy_dict[X[i]] = int(y[i])
    return xy_dict

        
def load_state_dict(model,fname):
    with open(fname,'rb') as f:
        weights = pickle.load(f,encoding='latin1')
        
    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError("Runtime Error")
        else:
            raise KeyError(f'Unexpected key {name} in state_dict')