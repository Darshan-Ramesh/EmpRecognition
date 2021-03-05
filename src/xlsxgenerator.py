import pandas as pd 
import numpy as np
import os
import glob as glob
import csv, json


class XlsxGenerator():
    """
    A class to generate excel file containing the emperor names and the image name read from a folder
    
    Attributes
    ----------
    rootfolder_path : str
      Parent folder path
    
    imagefolder_path : str
      Folder path where the images are present
      
    img_num : int
      Number to start from to rename the images inside a folder
  
    """
    def __init__(self,rootfolder_path,image_folder,img_num):
        self.rootfolder_path = rootfolder_path
        self.imagefolder_path = image_folder
        self.path = os.path.join(self.rootfolder_path + self.imagefolder_path)
        self.img_num = img_num


    def saveas_xlsx(self):
        """ Function that takes images path and creates a csv which includes image name (keizer (n).jpg) and emp name

        Arguments:
          mainfolder_path: path to the image folder
        """
        i = self.img_num
        count = 0
        #dict to hold the values. Key -> keizer (141).jpg, value-> image_name/emp_name
        names = {}
        for dirpath, dirname, filenames in os.walk(self.path):
            filenames.sort()
            for name in filenames:
                print(f'[INFO] Reading {self.path+name}')
                emp_name = self.get_name(name)
                img_name = "keizer "+ "(" + str(i) + ")" + ".jpg"
                print(f'[INFO] Changing {name} to {img_name}')
                os.rename(self.path+name,self.path+img_name)
                print('-'*100)
                i +=1
                names[img_name] = emp_name
        self.writeto_excel(self.path,names,excelname='data.xlsx')
        print(f'[INFO] Done!!')

    def get_name(self,filename):
        """ Discards the number in the image name

        Arguments:
          filename: Name of the file

        Returns:
          name of the file without numbers
        """
        name = "".join([i for i in (filename.split('.')[0]) if not i.isdigit()])
        if name[-2] == "(":
          name = name[:-3]
        return name

    @classmethod
    def writeto_excel(cls,path,names,excelname):
        """" writes the given dictionary to excel using pandas

        Arguments:
          names : Dictionary containing in {keizer (1).jpg : Caracalla} format

        """

        df = pd.DataFrame.from_dict(data={'ImageName': [i for i in names.keys()], 'EmpName': [i for i in names.values()]},orient='columns')
        print(f'[INFO] Save in - {path}')
        
        fullpath = os.path.join(path,excelname)
        if os.path.isfile(fullpath):
          os.remove(fullpath)

        writer = pd.ExcelWriter(path +'data.xlsx')
        # writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
        df.to_excel(writer,sheet_name='test1',index=False)
        writer.save()

 
if __name__ == "__main__":
  rootfolder_path="..\\Datasets"
  images_folder_path = "\\v4\\v4_4_testing\\onlyfaces\\"
  generator = XlsxGenerator(rootfolder_path,images_folder_path,514)
  generator.saveas_xlsx()