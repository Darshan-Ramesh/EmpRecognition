import os
import pickle
import torch


def get_cls_to_idx(datasetpath):
    """Method to map the foldernames/empnames into integers

    Args:
        datasetpath ([str]): Path to the folders whose names are of emperors, which contains images 

    Returns:
        [dict]: A dictionary of the mapping. Key is the Emperor name and value is associated id
    """
    i = 0
    map_dict = {}
    for foldernames in os.listdir(datasetpath):
        map_dict[foldernames] = i
        i += 1
    return map_dict


def get_idx_to_cls(cls_to_idx_dict):
    """Method to get the dict containing integer to Empname mapping

    Args:
        cls_to_idx_dict ([dict]): A dictionary of the mapping. Key is the Emperor name and value is associated id

    Returns:
        [dict]: Key is the integer and the value is the Emperor Name.
    """
    cls_to_idx = {v: k for (k, v) in cls_to_idx_dict.items()}

    return cls_to_idx


def create_dict(X, y):
    """Creates a dictinoary

    Args:
        X ([str]): full path to the image inside the emperor named folder
        y ([int]): Label of integer values 

    Returns:
        [dict]: Key: image path, value: label, integer values
    """
    assert len(X) == len(y)

    xy_dict = {}
    for i in range(len(X)):
        # emp_name = X[i].split('\\')[-2]
        xy_dict[X[i]] = int(y[i])
    return xy_dict


def load_state_dict(model, fname):
    """Standard function to load the pre-trained weights

    Args:
        model ([Ordererd dict]): model whose weights needs to be loaded
        fname ([str]): path to the pre-trained weights

    Raises:
        KeyError: Raise exception if the key value did not match with the constructed model
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError("Runtime Error")
        else:
            raise KeyError(f'Unexpected key {name} in state_dict')
