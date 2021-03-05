
import cv2 
import numpy as np
import imutils

from facedetector import FaceDetector
from imageutils import ImageUtils
import argparse



def manual_face_recog(path,img_size=256,save_in=None,resize=False,bgr2rgb=False,align=False):

    #read and conver to RGB
    print("-"*40)
    print(f'[INFO] Reading the image from  - {path}')
    img = cv2.imread(path)
    if resize:
        img = imutils.resize(img, width=800)
    if bgr2rgb:
        img_rgb =  cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    #get the points
    bbox = np.array(cv2.selectROI('Annotation',img_rgb))
    left_eye = np.array(cv2.selectROI('left eye',img_rgb))
    right_eye = np.array(cv2.selectROI('right eye',img_rgb))
    points = [[left_eye,right_eye]]
    print(points)
    print(f'[INFO] Face - {bbox},left_eye - {left_eye}, right_eye - {right_eye}')
    
    
    
    if align:
        #align the image
        print(f'[INFO] Aligning...')
        aligned_img = FaceDetector.alignment(points,img_rgb)
    else:
        aligned_img = img_rgb
    
    #crop the image with margin
    print(f'[INFO] Cropping...')
    boxes = [bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]]
    
    #make it square
    print(f'[INFO] Converting to square bboxes..')
    boxes = FaceDetector.make_square([boxes])
    print(f'[INFO] After converting - {boxes}')
    
    for box in boxes:
        cropped_face = FaceDetector.get_face(aligned_img,box,image_size=img_size,margin=0,save_path=None)
    
    #resize the image to 160x160
    print(f'[INFO] Resizing...')
    resized_face = ImageUtils.resize_pad(cropped_face,desired_size=img_size)
    
    #save
    if bgr2rgb:
        resized_face = cv2.cvtColor(resized_face,cv2.COLOR_RGB2BGR)
        
    print(f'[INFO] Saving the image in - {save_in}')
    cv2.imwrite(save_in,resized_face)
    print("-"*40)
    
    
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--imagename",required=False,help="Image name..eg: keizer (1).jpg")
    ap.add_argument("-s","--saveas",required=False,help='new image name if required..')
    args = vars(ap.parse_args())
    
    image_name = args["imagename"]
    save_as = args['saveas']
    # path = "..\\Datasets\\v4\\v4_2_testing\\" + str(image_name)
    # save_in = "..\\Datasets\\v4\\v4_2_testing\\onlyfaces\\" +  str(image_name)
    
    path = "..\\..\\Paper\\Images\\" + str(image_name)
    save_in = "..\\..\\Paper\\Images\\" + str(save_as)
    
    # path = "C:\\Users\\SrirangacharRamesD\\Pictures\\Unknown_Image_prediction\\1.jpg"
    # save_in = "C:\\Users\\SrirangacharRamesD\\Pictures\\Unknown_Image_prediction\\1_cropped.jpg"
    manual_face_recog(path,256,save_in,True,False,True)
    