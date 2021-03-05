import torch.nn as nn 
import torch
import pandas as pd 
import numpy as np 
import torch.nn.functional as F 
from sklearn.metrics import classification_report
import config as cfg
import empdataset
import matplotlib.pyplot as plt
from head import ArcFace, l2_norm
import seaborn as sns
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization,training
import imageutils
import argparse
from visualizer import draw_confusion_mat

def get_names(label,idx_to_class):
    names= [] 
    
    if isinstance(label,torch.Tensor):
        preds_ = np.array(label.cpu().flatten())
        for i in range(len(preds_)):
            names.append(idx_to_class[preds_[i]])
    else:
        preds_ = np.array(label)
        names.append(idx_to_class[label])
    
    return names

def get_probs(probabilities_tensor):
    prob =[]
    
    if isinstance(probabilities_tensor,torch.Tensor):
        probs_ = np.array(probabilities_tensor.cpu().detach().numpy().flatten())
        
        for i in range(len(probs_)):
            prob.append(probs_[i])
    else:
        probs_ = np.array(prob)
        prob.append(probs_)
        
    return prob

def predict_validation_images(model,model_head,loader,idx_to_class,device):
    y_hat = []
    yyy = []
    misclassified_dict = {}
    probabilities_dict = {}
    misclassified_img_path = []

    model.to(device)
    model.eval()
    
    if model_head:
        model_head.eval()
        model_head.to(device)   
        model.classify = False
    
    else:
        model.classify = True
        
    
    count_misclassified = 0
    count_images = 0
    
    
    for i, (im,label,img_path) in enumerate(loader):
        label = label.to(device)
        im = im.to(device)
        
        if model_head:
            with torch.no_grad():
                embeddings = model(im)
            y_pred, op_logits = model_head(embeddings,label)

        else:
            with torch.no_grad():
                op_logits = model(im)
            
        probs, preds = torch.topk(F.softmax(op_logits,dim=1),3)
    
        # predictions_array = np.array(preds.cpu().numpy().flatten())
        # y_hat.append(predictions_array[0])
        # yyy.append(label[0].cpu().numpy())
        
        for j in range(label.shape[0]):
            actual_name = get_names(label[j],idx_to_class)
            print(f'[INFO] Actual label - {actual_name}')
            
            predicted_names = get_names(preds[j],idx_to_class)
            print(f'[INFO] Predicted label - {predicted_names}')            

            y_hat.append(np.array(actual_name))
            yyy.append(np.array(predicted_names[0]))
            #print probs
            print(f'[INFO] Probabilities - {probs[j].detach().cpu().numpy().flatten()}')
            
            #count misclassified images
            misclassified = False if actual_name[0] == predicted_names[0] else True
            print(f'[INFO] Missclassified - {misclassified}')
            
            count_images += 1
            
            if misclassified:
                count_misclassified += 1
                print(f'[INFO Misclassified image path - {img_path[j]}')
                misclassified_img_path.append(img_path[j])
                
                misclassified_dict[img_path[j]] = predicted_names
                
                probabilities_dict[str(probs[j].detach().cpu().numpy().flatten())] = img_path[j]
                 
            print('-'*250)
    print('-'*250)
    print('[Report]')
    print('-'*250)
    assert count_misclassified == len(misclassified_img_path)
    print(f'[INFO] Misclassified - {count_misclassified}/{count_images}')
    print(f'[INFO] Misclassified images path - {misclassified_img_path}')

   
    
    #convert to df and save as excel
    if cfg.WRITE_TO_EXCEL:
        print('-'*250)
        print(f'[INFO] Generating excel sheet containing misclassified images and its predicted labels...')
        print('-'*250)
        
        df = pd.DataFrame.from_dict(data = {'ImagePath': [i for i in misclassified_dict.keys()],
                                            'Predicted Name': [i for i in misclassified_dict.values()]},
                                    orient='columns')
        df['Probabilties'] = probabilities_dict
        
        writer = pd.ExcelWriter(cfg.PREDICTION_LOGS +cfg.EXCEL_NAME)
        df.to_excel(writer,sheet_name='Misclassfied_Data',index=False)
        writer.save()
        print('-'*250)
        print(f'[INFO] Misclassified information is stored in - {cfg.PREDICTION_LOGS}')
        print('-'*250)

    y_hat = np.array(y_hat).flatten()
    yyy = np.array(yyy).flatten()
    
    # print(f'[INFO] y_hat shape - {y_hat.shape}')
    # print(f'[INFO] yyy shape - {yyy.shape}')
    
    # print(f'[INFO] Yhat - {y_hat}')
    # print(f'[INFO] Y_gt - {yyy}')    
    
    assert len(y_hat) == len(yyy)
    
    return y_hat,yyy

def print_classification_report(gt,y_hat,zero_divison=False):
    print(classification_report(gt,y_hat,zero_division=True)) 
    


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-s","--select",required=True,
                    help="1. Select validation set and 2. Select test set")
    args = vars(ap.parse_args())
    select = int(args['select'])
    
    #for debug
    # select = 1
    
    #load the model
    print(f'[INFO] Loading model from {cfg.LOAD_CHECKPOINT}')
    model = torch.load(cfg.LOAD_CHECKPOINT)
    print(model)
    
    if cfg.MODEL == 'Inception' or cfg.MODEL == 'resnet50':
        model.classify = True   

    # image = np.random.randn(160,160,3)
    # image = np.array(image,dtype=np.uint8)
    # image = imageutils.ImageUtils.apply_transforms(image)
    
    
    # logits = model(image.cuda())
    # print(logits.shape)
    # score,preds = torch.topk(F.softmax(logits,dim=1),3)
    # print(score)
    # print(preds)
    
    
    if cfg.HEAD:
        print(f'[INFO] Loading head from {cfg.LOAD_HEAD_CHECKPOINT}')
        
        model_head = torch.load(cfg.LOAD_HEAD_CHECKPOINT)
        print(f'[INFO] Loaded...')
    else:
        model_head = None
  
  
    valid_dataset = empdataset.EmpDataset(cfg.DATASET_PATH, cfg.VALID_EXCEL_PATH, augmentations=None)
    print(valid_dataset.idx_to_class)
    
    if select == 1:
        #define the valid_dataloader
    
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=1,
                                               shuffle=False, num_workers=0)
        #call prediction
        y_hat,y_gt = predict_validation_images(model,model_head,valid_loader,valid_dataset.idx_to_class,device=cfg.DEVICE)
        
        #print classification report
        print_classification_report(y_gt,y_hat)
        print(f'[INFO] Loaded model {cfg.LOAD_CHECKPOINT}')
        
        
        print('[INFO] Generating confusion matrix...')
        labels = [k for k,v in valid_dataset.cls_to_idx.items()]
        draw_confusion_mat(y_gt,y_hat,labels,normalized=None)
    
    if select == 2:
        
        test_dataset = empdataset.EmpDataset(cfg.DATASET_PATH,cfg.TEST_EXCEL_PATH,augmentations=None)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,
                                               shuffle=False, num_workers=0)

        #call prediction
        y_hat,y_gt = predict_validation_images(model,model_head,test_loader,valid_dataset.idx_to_class,device=cfg.DEVICE)
        
        #print classification report
        print_classification_report(y_gt,y_hat)
        print(f'[INFO] Loaded model {cfg.LOAD_CHECKPOINT}')
        
        
        # print('[INFO] Generating confusion matrix...')
        labels = [k for k,v in valid_dataset.cls_to_idx.items()]    
        draw_confusion_mat(y_gt,y_hat,labels,normalized=None)
    