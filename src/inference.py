import torch.nn as nn 
import torch
import pandas as pd 
import numpy as np 
import torch.nn.functional as F 
from sklearn.metrics import classification_report, confusion_matrix
import config as cfg
import empdataset
import matplotlib.pyplot as plt
from head import ArcFace, l2_norm
import seaborn as sns
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization,training
import imageutils
import argparse
from PIL import Image
import cv2
from torchvision import models

    
IDX_TO_CLASS = {0: 'Antoninus Pius',
                1: 'Augustus',
                2: 'Caracalla',
                3: 'Commodus',
                4: 'Hadrian',
                5: 'Lucius Verus',
                6: 'Marcus Aurelius',
                7: 'Septimius Severus',
                8: 'Trajan',
                9: 'NonEmperors'}


def infer(image,format_image=False,emp_name=None):
    
    if format_image:
        image = np.array(image,dtype=np.uint8)
        image = imageutils.ImageUtils.apply_transforms(image,model=cfg.MODEL)
        print(image.shape)
    
    print(f'[INFO] Loading model from - {cfg.LOAD_CHECKPOINT}')
    model = torch.load(cfg.LOAD_CHECKPOINT)
    model.eval()
    model.cuda()
    
    # for p in model.parameters():
    #     print(p.requires_grad)
    
    with torch.no_grad():
        logits = model(image.cuda())
    print(f'[INFO] Shape of logits - {logits.shape}')
    scores,preds = torch.topk(F.softmax(logits,dim=1),3)
    wo_softmax,_ = torch.topk(logits,3)
    scores = scores.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()[0]
    
    print('\n')
    print('*'*100)    
    print(f'[INFO] Pre Softmax Scores = {wo_softmax.cpu().numpy()[0]}\n')
    print(f'[INFO] Post Softmax Scores - {scores[0]}\n')
    
    if emp_name is not None:
        print(f'[INFO] Actual name - {emp_name}')
    print(f'[INFO] Predicted names - {[IDX_TO_CLASS[i] for i in preds]}')
    print(f'[INFO] Predicted labels - {preds}')
    print('*'*100)




if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--imagepath",required=True,
                    help="Input the full image path")
    args = vars(ap.parse_args())
    
    image_path = args['imagepath']
    image = Image.open(image_path).convert('RGB')
    
    # random nose
    # image = np.random.randn(160,160,3)
    
    plt.imshow(image)
    plt.show()
    
    if cv2.waitKey() == ord('q'):
        exit

    infer(image,True)
    
    
    
    
    
    