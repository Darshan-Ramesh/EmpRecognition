import os
import numpy as np 
import pandas as pd 
from albumentations import (Compose,OneOf,RandomBrightnessContrast,
                            RandomGamma,ShiftScaleRotate,HorizontalFlip,
                            Rotate,FancyPCA,RandomCrop,RandomBrightnessContrast,ToGray,
                            MultiplicativeNoise)
import empdataset
import engine
import prediction_report
from lossfunctions import LossFunctions
from models import get_model
import cv2
import matplotlib.pyplot as plt
from glob import glob
import torchvision
import torch
from torchvision import models,datasets,transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torchsummary import summary
from PIL import Image,ImageShow
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization,training
import pickle
from shutil import copyfile
from head import ArcFace
from torch.cuda import amp

import warnings
warnings.filterwarnings('ignore')
import config as cfg


def df_check(df_train,df_valid):
    print('*'*50)
    print(f'[INFO] Checking Dataframes for missing emp names in validation set..')
    print('*'*50)
    
    diff_empnames = set(df_train[df_train.columns[1]]) - set(df_valid[df_valid.columns[1]])
    if diff_empnames is not None:
        print(f'[INFO] Validation dataset has no - {diff_empnames}  emperors from training set')
    else:
        print(f'[INFO] Validation dataset has no - {diff_empnames}  emperors from training set- Stratified')

    print('-'*50)
    print('\n')
    
    print('*'*50)
    print(f'[INFO] Comparing Dataframes for data leaks from training set to validation set...')
    data_leak_check = set(df_train.ImagePath).intersection(set(df_valid.ImagePath))
    if not data_leak_check:
        print(f'[INFO] No data leak from validation set to training set...')
    else:
        print(f'[INFO] Data leak from validation set to training set... - {data_leak_check}')
        print('Breaking..')
        exit
        
    print('*'*50)
    

def load_datasets(dataset_path,excel_path,augmentations=None,training_data=False):
    #train and validation test
    print('*'*50)
    if training_data:
        print(f'[INFO] Train Dataset information')
    else:
        print(f'[INFO] Validation Dataset information')
    print('*'*50)
    dataset = empdataset.EmpDataset(dataset_path,excel_path,augmentations=augmentations)
    dataset.print_info()
    print('-'*50)
    print('\n')
    
    return dataset
    
def get_cls_weights(samples_per_cls,effective_weights=False):
    
    if effective_weights: #from Class balance loss based on Effective number of samples paper
        effective_num = 1.0 - np.power(cfg.BETA,samples_per_cls)
        weights = (1.0-cfg.BETA)/np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_cls)
        
    else:
        #smooth version
         weights = 1./(samples_per_cls)
    
    weights = torch.FloatTensor(weights).to(cfg.DEVICE)
    return weights

def get_samples_per_cls(y):
    class_sample_count = np.array([len(np.where(y==t)[0]) for t in np.unique(y)])
    return class_sample_count

def cp_and_convert(config_file,save_in):
    filename = config_file.split('.')[0] + '.txt'
    copyfile(src=config_file,
             dst=save_in+filename)


def separate_bn(model):
    all_params = model.parameters()
    paras_only_bn = []
    for pname,p in model.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
    paras_only_bn_id = list(map(id,paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id,all_params))
    return paras_only_bn, paras_wo_bn
            
def get_weighted_sampler(cls_sample_count,y):
    weights = 1./cls_sample_count
    samples_weight = torch.from_numpy(np.array([weights[t] for t in y]))
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),len(samples_weight))
    
    return sampler
    
            
if __name__ == '__main__':
    
    ##################################################### DATASET RELATED #####################################################
    ###########################################################################################################################
    
    #set the seed for reproducing same results on every run
    #from https://github.com/HuangYG123/CurricularFace/blob/master/train.py (line 44-46)
    #this might increase the training time per epoch, but gurantees reproducibility
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    train_excel_path = cfg.TRAIN_EXCEL_PATH
    valid_excel_path = cfg.VALID_EXCEL_PATH
    dataset_path = cfg.DATASET_PATH
     
    device = cfg.DEVICE
    print(f'[INFO Will be training on {device}...')

    if cfg.AUGMENTATION:
        #augmentations
        aug = Compose([HorizontalFlip(p=0.5),
                       ShiftScaleRotate(shift_limit=(0.02),
                                scale_limit=(0.2),
                                rotate_limit=10,p=0.8),
                       RandomCrop(224,224,p=0.8),
                       RandomBrightnessContrast(p=0.8),
                       MultiplicativeNoise(multiplier=[0.8, 1], elementwise=True, per_channel=True, p=0.8)
                    ])
    else:
        aug = None
    
    #datasets
    train_dataset = load_datasets(dataset_path,train_excel_path,aug,training_data=True)
    valid_dataset = load_datasets(dataset_path,valid_excel_path,None,training_data=False)
    
    #dataframes check
    df_check(train_dataset.df,valid_dataset.df)
    
    ##################################################### MODEL & HEAD RELATED ######################################################
    ##########################################################################################################################
    model,model_head = get_model(model_name = cfg.MODEL)
    model.to(device)
    
    if model_head:
        model_head.to(device)
    
    # model_head = cfg.HEAD
    
    ##################################################### CLASS WEIGHTS RELATED ###############################################
    ###########################################################################################################################
    #get sample counts of each class
    cols = cfg.COLUMNS 
    train_samples_count = get_samples_per_cls(train_dataset.df[cols[1]]) #column = label from the train.xlsx
    valid_samples_count = get_samples_per_cls(valid_dataset.df[cols[1]]) #column = label from the valid.xlsx
    
    #define class_weights
    #1. cls_weights = 1/np.sqrt(class_sample_counts) -> smooth version
    #2. cls_weights = effective_cls_weights from the paper https://arxiv.org/abs/1901.05555

    valid_cls_weights = get_cls_weights(valid_samples_count,effective_weights=True)
    train_cls_weights = get_cls_weights(train_samples_count,effective_weights=True)
 
    
    ##################################################### DATALOADERS #########################################################
    ###########################################################################################################################
    print('*'*100)        
    
    if cfg.WEIGHTED_SAMPLING:
        print(f'[INFO] Using Weighted sampling...')
        weighted_sampler = get_weighted_sampler(train_samples_count,train_dataset.df[cols[1]])
    
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.TRAIN_BATCH_SIZE,
                                               shuffle=False,num_workers=0,sampler=weighted_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=cfg.TRAIN_BATCH_SIZE,
                                               shuffle=True,num_workers=0,sampler=None)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=cfg.VALID_BATCH_SIZE,
                                               shuffle=False,num_workers=0)
    
    #define loss functions
    train_loss_fn = LossFunctions(loss_fn_name=cfg.LOSS_FN,alpha=1,gamma=cfg.GAMMA,cls_weights=None)
    valid_loss_fn = LossFunctions(loss_fn_name=cfg.LOSS_FN,alpha=1,gamma=cfg.GAMMA,cls_weights=None)
    
    
    ##################################################### OPTIMIZERS #########################################################
    ###########################################################################################################################
    # define optimizer
    # optimizer = torch.optim.Adam(model.logits.parameters(),lr=cfg.LR_RATE)
    if cfg.HEAD:
        print(f'[INFO] IN head!')
        
        if cfg.MODEL == 'resnet34' or cfg.MODEL == 'Inception' or cfg.MODEL == 'resnet50' \
                                   or cfg.MODEL == 'IR50' :
                                       
            print(f'[INFO] Separating Batch Normalizations for optimizer...')
            with_bn,without_bn = separate_bn(model)
        else:
            model_params = filter(lambda p: p.requires_grad, model.parameters())
            
        model_head_params = list(model_head.parameters())
        
        print(f'[INFO] Optimizer - {cfg.OPTIMIZER}')
        print('*'*100)        
        
        if cfg.OPTIMIZER == 'ADAMW':
            # print(f'[INFO] Using - {cfg.OPTIMIZER} as optimizer..')
            # optimizer = torch.optim.Adam([{'params': model_params, 'weight_decay': cfg.W_DECAY },
            #                               {'params': model_head_params, 'weight_decay': cfg.W_DECAY}], 
            #                              lr=cfg.LR_RATE)
            optimizer = torch.optim.AdamW([{'params': without_bn + model_head_params,
                                           'weight_decay': cfg.W_DECAY},
                                          {'params': with_bn}], lr=cfg.LR_RATE)

        elif cfg.OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam([{'params': without_bn + model_head_params,
                                           'weight_decay': cfg.W_DECAY},
                                            {'params': with_bn}], lr=cfg.LR_RATE)
        
        elif cfg.OPTIMIZER == 'SGD':
            # optimizer = torch.optim.SGD([{'params': model_params, 'weight_decay': cfg.W_DECAY},
            #                              {'params': model_head_params, 'weight_decay': cfg.W_DECAY}], 
            #                             lr=cfg.LR_RATE,momentum=0.9)
            optimizer = torch.optim.SGD([{'params': without_bn + model_head_params,
                                           'weight_decay': cfg.W_DECAY},
                                          {'params': with_bn}], 
                                        lr=cfg.LR_RATE,momentum=0.9)
    else:
        # print(f'[INFO] Separating Batch Normalizations for optimizer...')
        with_bn,without_bn = separate_bn(model)
        params_without_bn = filter(lambda p: p.requires_grad, without_bn)
        params_with_bn = filter(lambda a:a.requires_grad,with_bn)

        if cfg.OPTIMIZER == 'ADAM':
            optimizer = torch.optim.Adam([{'params': params_with_bn},
                                           {'params': params_without_bn,'weight_decay':cfg.W_DECAY}],
                                          lr = cfg.LR_RATE)
        
        if cfg.OPTIMIZER == 'ADAMW':
            optimizer = torch.optim.AdamW([{'params': params_with_bn},
                                           {'params': params_without_bn,'weight_decay':cfg.W_DECAY}],
                                          lr = cfg.LR_RATE)
        
        elif cfg.OPTIMIZER == 'SGD':
            optimizer = torch.optim.SGD([{'params': without_bn,
                                           'weight_decay': cfg.W_DECAY},
                                          {'params': params_with_bn}], 
                                        lr=cfg.LR_RATE,momentum=0.9)
            
    #epochs
    epochs = cfg.EPOCHS
    
    #define scheduler
    scheduler = MultiStepLR(optimizer,cfg.STEPS)
    
    #define metrics to track
    # supports 'precision,recall and f1score'
    metrics = {'Accuracy': engine.accuracy,
               'F1 Score': engine.f1score,
               'Batch Time': engine.BatchTimer()
               }
    
    #define Summary writer
    writer = SummaryWriter(log_dir=cfg.RUNS)
    writer.iteration, writer.interval = 0,63
    
    ############################### CONFIGURATIONS##############################################################################
    ############################################################################################################################
    #TODO:
    # 1. copy the config.py of every experiment and rename to .txt file
    print(f'[INFO] Copying the configuration files to the logs folder...')
    files = ['config.py','models.py','train.py']
    for f in files:
        cp_and_convert(config_file=f,
                   save_in= cfg.RUNS)
    
    #using automatic mixed precision training
    scaler = amp.GradScaler()
    
    #train and evaluation
    print('\n\nInitializing..')
    print('-'*30)

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch+1,epochs))
        print('-'*30)

        model.train()
        # model_head.train()
        engine.pass_epoch(model,model_head,train_loss_fn,train_loader,optimizer,scheduler,metrics,
                          show_running=True,device=device,writer=writer,epoch=epoch,grad_scaler = scaler)

        with torch.no_grad():
            model.eval()
            
            if model_head:
                model_head.eval()
            
            engine.pass_epoch(model,model_head,valid_loss_fn,valid_loader,optimizer,scheduler,metrics,
                              show_running=True,device=device,writer=writer,epoch=epoch,grad_scaler = None)

    #save the model
        model_path = os.path.join(cfg.MODEL_BASE_FOLDER, cfg.PATH)
        if not os.path.exists(model_path):
            print(f'[INFO] Folder does not exist..creating one..')
            os.mkdir(model_path)
        
        if cfg.SAVE_MODEL:
            #SAve the model    
            save_as = str(cfg.SAVE_CHECKPOINT)
            torch.save(model,save_as)
            
        
        if cfg.HEAD is not None and cfg.SAVE_HEAD:
            #Save the head params
            save_head_as = str(cfg.SAVE_CHECKPOINT_HEAD)
            torch.save(model_head,save_head_as)
        
    
    writer.close()
    torch.cuda.empty_cache()
    
    
    