import cv2 
import numpy as np 
import torch 
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn 
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFile
import pandas as pd 
from  dict_mappings import get_idx_to_cls, get_cls_to_idx, create_dict
import config as cfg
import albumentations
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

class EmpDataset(Dataset):
    def __init__(self,dataset_path,excel_path,augmentations=None):
        self.df = pd.read_excel(excel_path)
        self.columns = self.df.columns
        self.datasetpath = dataset_path
        
        #first column is the image path
        #second column is the integer of (label)
        #third column is strings of emperor names
        self.image_paths = np.array(self.df[self.columns[0]])
        self.targets = self.df[self.columns[1]]
        self.empnames = (self.df[self.columns[2]].unique())
        self.classes = self.targets.unique()
        
        self.cls_to_idx = dict(zip(self.empnames,self.classes))
        self.idx_to_class = dict(zip(self.classes,self.empnames))
        
        self.augmentations = augmentations
    
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,index):
        image_paths = self.image_paths[index]
        image = Image.open(image_paths).convert('RGB')
        
        targets = self.targets[index]
        targets = torch.tensor(targets,dtype=torch.long)
        
        #for hair clasification
        # image = image.resize((224,224))
        
        #convert to Numpy array
        image = np.array(image)
        image = albumentations.Resize(height=224,width=224,interpolation=3,always_apply=True)(image=image)['image']
        
        # #if using ir_50 model -> input size -> [112x112]
        if cfg.MODEL == 'IR50':
            image = image.resize((112,112))
        
        if self.augmentations is not None:

            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        
        if cfg.MODEL == 'Inception':            
            #resize to 160x160
            image = albumentations.Resize(height=160,width=160,interpolation=3,always_apply=True)(image=image)
            image = image['image']
            assert image.shape == (160,160,3)
            
            #HxWxC -> CxHxW
            image = np.transpose(image, (2,0,1)).astype(np.float32)
            
            #convert to tensor -> [min,max] -> [0,255]
            image = torch.tensor(image,dtype=torch.float)
            
            #standadization [min,max] -> [0,1]
            image = image_standardization(image)
            # assert image.min() <= 0 
            # assert image.max() <= 1
            
        elif cfg.MODEL == 'resnet50' or cfg.MODEL == 'senet50':
            #image shape is 224x224
            assert image.shape == (224,224,3)
            
            image = transform(image)
            
        #IR50
        elif cfg.MODEL == 'IR50':
            #[0,255] -> [-1,1]
            image = transform_ir(image)
            assert image.min() < 0
            assert image.max() <= 1
        
        elif cfg.MODEL == 'resnet34' or cfg.MODEL == 'resnet18':
            to_tensor = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
            image = to_tensor(image)
        
        
        return image, targets, image_paths
        
    def print_info(self):
        print(f'[INFO] We have columns - {self.columns}')
        print(f'[INFO] Dataset length - {len(self.image_paths)}')
        print(f'[INFO] Target length - {len(self.targets)}')

def image_standardization(image_tensor):
    image_tensor = (image_tensor - 127.5)/128.0
    return image_tensor

def transform(img):
    mean_rgb = np.array([131.0912,103.8827,91.4953])
    #from  https://github.com/cydonia999/VGGFace2-pytorch/blob/master/datasets/vgg_face2.py
    # img = img[:,:,::-1] #-> RGB to BGR
    img = img.astype(np.float32)
    img -= mean_rgb
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

def transform_ir(img):
    # RGB_MEAN = (0.5,0.5,0.5)
    # RGB_STD= (0.5,0.5,0.5)
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    img = trans(img)
    
    return img
   