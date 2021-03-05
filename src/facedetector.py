import cv2
import numpy as np
import pandas as pd
import os
import glob as glob
import matplotlib.pyplot as plt
import csv
import json
import shutil
import seaborn as sns
from matplotlib.pyplot import figure, show
from glob import glob
import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from imageutils import ImageUtils


class FaceDetector():
    def __init__(self, datapath, save_path, image_size=256, align=False, device='cpu'):
        self.detector = MTCNN(image_size=160, margin=20,
                              post_process=False, min_face_size=10, device=device)
        self.path = datapath
        self.save_path = save_path
        self.align = align
        self.image_size = image_size
        print(f'[INFO] Dataset path - {self.path}')
        print(f'[INFO] Saving path - {self.save_path}')

    def detect_faces(self):
        noface_path = []
        extension = "*.tiff"
        resized_face = []
        # print(f'[INFO] Reading - {self.path}')
        # print(f'[INFO] Detecting faces...')
        # save_path = 'onlyfaces\\bynames_mtcnn_2'
        for folder_path in glob(self.path):
            print(folder_path)
            for path in glob(os.path.join(folder_path, extension)):
                print('-'*30)
                print(f'[INFO] Reading from -{path}')
                image = cv2.imread(path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image_ = self.detector(image_rgb)
                boxes, probs, points = self.detector.detect(
                    image_rgb, landmarks=True)

                if boxes is None:
                    print(f'[INFO] Faces not found in {path}')
                    noface_path.append(path)

                else:
                    print(f'[INFO] Found face..')

                    print(f'[INFO] Making the boxes square..')
                    boxes = self.make_square(boxes)

                    if self.align:
                        print('[INFO] Aligning...')
                        image_rgb = self.alignment(points, image_rgb)

                    for i, (box, point) in enumerate(zip(boxes, points)):
                        face = self.get_face(
                            image_rgb, box, self.image_size, 0, None)
                        resized_face = ImageUtils.resize_pad(
                            face, self.image_size)
                        resized_face = cv2.cvtColor(
                            resized_face, cv2.COLOR_RGB2BGR)

                    # image_face = image_.numpy().transpose(1,2,0).astype('uint8')
                    new_paths = path.split('\\')[-2:]
                    foldername = new_paths[0]
                    filename = new_paths[1].split('.')[0] + '.jpg'

                    # save_in = os.path.join(self.save_path,foldername)
                    if not os.path.isdir(self.save_path):
                        os.mkdir(self.save_path)

                    print(
                        f'[INFO] Saving in - {os.path.join(self.save_path,filename)}')
                    cv2.imwrite(os.path.join(
                        self.save_path, filename), resized_face)
                    print('-'*30)
        print('-'*30)
        print(f'[INFO] No faces detectes images number - {len(noface_path)}')
        print(f'[INFO] No face images list - {noface_path}')
        print('-'*30)

    @classmethod
    def alignment(cls, points, img):
        left_eye = points[0][0]
        right_eye = points[0][1]

        x1, y1 = left_eye[0], left_eye[1]
        x2, y2 = right_eye[0], right_eye[1]
        tan = (y2-y1)/(x2-x1)
        angles = np.degrees(np.arctan(tan))
        print(f'[INFO] Angles is {angles}')

        xc = (x1+x2)/2
        yc = (y1+y2)/2
        M = cv2.getRotationMatrix2D((xc, yc), angles, 1)
        # M = cv2.estimateAffine2D()

        img = np.array(img)
        rotated_img = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        return rotated_img

    @classmethod
    def get_face(cls, img, box, image_size=256, margin=0, save_path=None):
        if margin != 0:
            margin = [
                margin * (box[2] - box[0]) / (image_size - margin),
                margin * (box[3] - box[1]) / (image_size - margin),
            ]
            print(f'[INFO] Margin values - {margin}')

            if isinstance(img, (np.ndarray, torch.Tensor)):
                raw_image_size = img.shape[1::-1]

            else:
                raw_image_size = img.size

            print(f'[INFO] Box before adding margin values - {box}')

            box = [
                int(max(box[0] - margin[0] / 2, 0)),
                int(max(box[1] - margin[1] / 2, 0)),
                int(min(box[2] + margin[0] / 2, raw_image_size[0])),
                int(min(box[3] + margin[1] / 2, raw_image_size[1])),
            ]
            print(f'[INFO] New boxes after adding margin values - {box}')
            img = img[box[1]:box[3], box[0]:box[2]]
        else:
            img = img[box[1]:box[3], box[0]:box[2]]
        return img

    @classmethod
    def make_square(cls, boxes):
        boxes = boxes
        x = boxes[0][0]
        y = boxes[0][1]
        w = boxes[0][2] - x
        h = boxes[0][3] - y

        # cropping used from https://gist.github.com/tilfin/98bbba47fdc4ac10c4069cce5fabd834
        r = max(h, w)/2
        centerX = x + w/2
        centerY = y + h/2
        nx = int(centerX - r)
        if nx <= 0:
            nx = 0
        ny = int(centerY - r)
        nr = int(r*2)

        new_boxes = [[nx, ny, nx+nr, ny+nr]]
        return new_boxes


# TODO
# 1. Changing the 'min_face_size' when no face is detected. Increasing the size/decreasing the size usually works
# 2. Handle when there are two boxes get detected
# 3. When there is no detection at all after Step 1, do it manually.
if __name__ == "__main__":
    # dataset_path = "..\\Datasets\\v5\\v5_3_testing\\"
    # save_in = "..\\Datasets\\v5\\v5_3_testing\\onlyfaces\\"

    dataset_path = "..\\Datasets\\v6\\Emperors\\AntoninusPius\\"
    save_in = "..\\Datasets\\v6\\Emperors\\AntoninusPius\\onlyfaces\\"

    if not os.path.isdir(save_in):
        os.mkdir(save_in)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = FaceDetector(dataset_path, save_in, 256, True, device)
    mtcnn.detect_faces()
