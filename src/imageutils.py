import cv2
import glob as glob
from torchvision import transforms
import numpy as np 
from empdataset import image_standardization
import torch
from albumentations import Resize
from torch.autograd import Variable

class ImageUtils():
    def __init__(self):
        pass
        
    def modify_xywh(self,face,image):
        dx = x - int(x*0.25)
        dy = y - int(y*0.25)
        dw = w + int(w*.1)
        dh = h + int(h*.08)
        (x,y,w,h) = face['box']
        return (dx,dy,dw,dh)
    
    @classmethod
    def resize_pad(cls,image,desired_size=160):
        old_size = image.shape
        print(f'[INFO] Original size - {old_size}')
        
        ratio = float(desired_size)/max(old_size)
        print(f'[INFO] ratio is - {ratio}')
        
        new_size = tuple([int(x*ratio) for x in old_size])
        print(f'[INFO] New size - {new_size}')
        
        resized_im = cv2.resize(image,(new_size[1],new_size[0]),interpolation=cv2.INTER_AREA)
        print(f'[INFO] REsized size - {resized_im.shape}')
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        print(f'[INFO] delta w - {delta_w}')
        print(f'[INFO] delta h - {delta_h}')
        
        top,bottom = delta_h//2 , delta_h - (delta_h//2)
        left,right = delta_w//2 , delta_w - (delta_w//2)
        color = [0, 0, 0]
        
        new_im = cv2.copyMakeBorder(resized_im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        print(f'[INFO] After padding - {new_im.shape}')
        return new_im

    def save_as(self,path):
        filename = path.split('\\')[-2:]
        return filename
    
    @classmethod
    def apply_transforms(cls,image,model='Inception',normalize=True):
        
        if model == 'Inception':
            # image = transforms.Resize(224)(image)
        #     image = transforms.CenterCrop(224)(image)
            if normalize:
                
                #conver to numpy
                im_arr = np.array(image, dtype = np.uint8)
                
                im_arr = Resize(height = 224,width = 224, interpolation = 3, always_apply = True)(image=im_arr)['image']
                im_arr = Resize(height = 160,width = 160, interpolation = 3, always_apply = True)(image=im_arr)['image']
                        
                #HxWxC -> CxHxW
                im_arr = np.transpose(im_arr, (2,0,1)).astype(np.float32)
                
                #tensor
                im_tensor = torch.tensor(im_arr,dtype=torch.float)
                print(f'[INFO] Tensor conversion (min,max) - {im_tensor.min(),im_tensor.max()}')
                
                #For inception
                im_tensor = image_standardization(im_tensor)
                print(f'[INFO] Post image standardization (min,max) - {im_tensor.min(),im_tensor.max()}')
                
                #add the extra dimension
                im_tensor = im_tensor.unsqueeze(0)
                
                # im_tensor = Variable(im_tensor, requires_grad = True)
                im_tensor.requires_grad = True
                
                return im_tensor

            else:
                
                img = image.squeeze(0)
                image = img.detach().cpu().numpy()
                
                image *= 128.0
                image += 127.5
                
                image = image.transpose((1,2,0)).astype('uint8')
                
                return image
        
        if model == 'resnet50' or model=='senet50':
            
            mean_rgb = np.array([131.0912,103.8827,91.4953])
            
            if normalize:
                im_arr = np.array(image, dtype = np.uint8)
                img_arr = Resize(height = 224,width = 224, interpolation = 3, always_apply = True)(image=im_arr)
                img = img_arr['image']
                img = img.astype(np.float32)
                print(f'[INFO] Pre normalization values - (min,max) - {img.min(),img.max()}')
                
                #from  https://github.com/cydonia999/VGGFace2-pytorch/blob/master/datasets/vgg_face2.py
                img = img[:,:,::-1] #-> RGB to BGR
                img -= mean_rgb
                img = img.transpose(2, 0, 1)  # C x H x W
                img = torch.from_numpy(img).float()     
                img = img.unsqueeze(0) 
                print(f'[INFO] Post normalization values - (min,max) - {img.min(),img.max()}')

                return img

            else:
                img  = image.squeeze(0).detach().cpu().numpy()
                img = img.transpose((1,2,0))
                img += mean_rgb
                
                return img.astype(np.uint8)
                
                
                
                
                
                
                
            
        
        