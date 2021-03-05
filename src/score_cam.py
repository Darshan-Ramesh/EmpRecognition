import numpy as np 
import torch
import torch.nn.functional as F 
import argparse
from PIL import Image
import pandas as pd
import config as cfg
import matplotlib.pyplot as plt
from imageutils import ImageUtils
from inference import infer
import matplotlib.cm as mpl_color_map
import copy
import tqdm
from torch.autograd import Variable
import cv2
from torchvision import transforms


class ScoreCam():
    def __init__(self,model,model_name):
        self.model = model
        self.model_name = model_name
        self.activations = {}
        
    def generate_cam(self,input_image,label,target_class=None):
        
        input_image = input_image.cuda()
        input_image = Variable(input_image)
        label = label.cuda()
        
        if self.model_name == 'Inception':
        #register the hook on a conv layer
            target_layer_name = 'block8.conv2d'
            self.model.block8.conv2d.register_forward_hook(self.get_activations(target_layer_name))
            logits = self.model(input_image)
            conv_op = self.activations[target_layer_name].squeeze()
            
            target = conv_op.clone()
            target = target.detach().cpu().numpy()
            print(f'[INFO] Shape of target - {target.shape}')
            
            if target_class is None:
                target_class = np.argmax(logits.detach().cpu().numpy())
                print(f'[INFO] Target class is - {target_class}')
            
            cam = np.ones(target.shape[1:],dtype=np.float32)
            count = 0
            
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(target))):
                    saliency_map = torch.unsqueeze(torch.unsqueeze(conv_op[i,:,:],0),0)
                    
                    #upsampling
                    saliency_map = F.interpolate(saliency_map,size=(input_image.size(2),input_image.size(3)),
                                                 mode='bilinear',align_corners=False)
                    
                    if saliency_map.max() == saliency_map.min():
                        count +=1
                        continue
                    
                    #norm btw 0 and 1
                    # norm_saliency_map = (saliency_map - saliency_map.min())/(saliency_map.max() - saliency_map.min())
                    norm_saliency_map = saliency_map / saliency_map.max()
                   
                    #get score
                    w = F.softmax(self.model(input_image * norm_saliency_map.cuda()),dim=1)[0][target_class]
                    
                    cam += w.data.cpu().numpy() * target[i,:,:]
                    
            cam = np.maximum(cam,0)
            cam = (cam - np.min(cam) / (np.max(cam) - np.min(cam)))
            
            cam = np.uint8(cam*255)
            cam = np.uint8(Image.fromarray(cam).resize((input_image.size(2),input_image.size(3)),Image.ANTIALIAS))/255
            return cam,target
            
        
    def get_activations(self,name):
        def hook(model,input,output):
            self.activations[name] = output
        return hook
        

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i","--index",required=True,
    #                 help="Input the image path index from excel sheet")
    # args = vars(ap.parse_args())
    
    # index = int(args['index'])
    index = 11
    df = pd.read_excel(cfg.VALID_EXCEL_PATH)
    path = df.ImagePath[index]
    label = df.Label[index]
    emp_name = df.EmpName[index]
    
    print(f'[INFO] Loading image from - {path}')
    print(f'[INFO] Loading model.. from - {cfg.LOAD_CHECKPOINT}')
    model = torch.load(cfg.LOAD_CHECKPOINT)
    
    #define transformers
    
    image = Image.open(path).convert('RGB')
    # image = np.array(image,dtype=np.uint8)
    # image = ImageUtils.apply_transforms(image,normalize=True)
    trans  = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
    ])
    image = trans(image)
    image = image/image.max()
    image = image.unsqueeze(0)
    
    print('-'*60)
    print('Inference')
    print('-'*60)
    infer(image,format_image=False,emp_name = emp_name)
    print('-'*60)
    
    print('Class Activation maps')

    
    score_cam = ScoreCam(model,model_name='Inception')
    cam,target = score_cam.generate_cam(image,torch.LongTensor(label))
    
    cam_maps = []
    print(f'[INFO] Loading image from - {path}')
    img = Image.open(path).convert('RGB')
    img = img.resize((160,160),resample=3)
    cam_maps.append(img)

    color_map = mpl_color_map.get_cmap('brg')
    no_trans_heatmap = color_map(cam)

    #change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:,:,3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    #apply heatmap on image
    heatmap_on_image = Image.new("RGBA",img.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image,img.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image,heatmap)

    cam_maps.append(heatmap_on_image)
    cam_maps.append(cam)

    plt.figure(figsize=(20,20))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(cam_maps[i])
    plt.show()
    
    
    
    
    
        