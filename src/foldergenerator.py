import cv2
import numpy as np
import pandas as pd
import os
import glob as glob
import matplotlib.pyplot as plt
import csv, json
import shutil


class FolderGenerator():
  def __init__(self,mainpath,images_folder_path,excel_path):
    self.mainpath = mainpath                            # "/content/drive/My Drive/Datasets/RomanEmperors/"
    self.images_folder_path = images_folder_path        # "newimages/"
    self.path = os.path.join(self.mainpath, self.images_folder_path)   # ""/content/drive/My Drive/Datasets/RomanEmperors/newimages"
    self.excel_path = excel_path
    
    self.df = pd.read_excel(self.excel_path)
    self.folder_count = self.df_foldercount()
    self.emp_names = self.get_empnames()
    self.mapping_dict = self.empname_imgname_mapping()


  def df_foldercount(self):
    cols = self.df.columns
    print(f'[INFO] Cols are - {cols}')
    count = len(self.df[cols[1]].unique())
    print(f'[INFO] We have {count} unique emp names! {count} emp folders will be created!')
    return count

  def get_empnames(self):
    cols = self.df.columns
    emp_names = self.df[cols[1]].unique()
    print(f'[INFO] Emp names are - {emp_names}')
    return emp_names

  def empname_imgname_mapping(self):
    emp_img_map = {}
    for i in range(len(self.emp_names)):
      name = self.emp_names[i]
      print(f'[INFO] Mapping {name,i} ')
      emp_img_map[name] = self.path + self.get_im_names(name)
    return emp_img_map

  def get_im_names(self,name):
    cols = self.df.columns
    names_df = self.df[self.df[cols[1]]==name]
    return np.array(names_df[cols[0]])

  def crtfolders(self,save_in):
    for name in self.emp_names:
      print(f'[INFO] Creating {name} folder')
      dir = os.path.join(save_in,name)
      if not os.path.exists(dir):
        os.mkdir(dir)

  def copy_images(self,save_in):
    for emp_name,img_path in self.mapping_dict.items():
      directory = os.path.join(save_in,emp_name)
      print(f'[INFO] checking if {directory} exists...\n')
      if os.path.exists(directory):
        print(f'[INFO]Pass..Copying images to folder\n')
        for f in img_path:
          print(f'[INFO] Copying {f}')
          shutil.copy(f,directory)


if __name__ == "__main__":
    dataset_path = "..\\Datasets\\"
    folder_name = "v5\\keizer_v5\\onlyhair\\"
    excel_path = "v5_onlyhair.xlsx"
    excel_path = os.path.join(dataset_path,folder_name,excel_path)
    
    save_folder = "bynames\\"
    save_in = os.path.join(dataset_path,folder_name,save_folder)
    
    if not os.path.exists(save_in):
      print(f'[INFO] Folder does not exists..creating one..')
      os.mkdir(save_in)
      
    
    test = FolderGenerator(dataset_path,folder_name,excel_path)
    
    test.crtfolders(save_in)
    
    test.copy_images(save_in)