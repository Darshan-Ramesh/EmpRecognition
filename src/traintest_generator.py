import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import train_test_split
from dict_mappings import get_idx_to_cls, get_cls_to_idx, create_dict
from xlsxgenerator import XlsxGenerator
import config as cfg

EXTENSION = "*.jpg"


class TrainTest_Generator:
    """Class that generates two excel sheets split in a stratified manner to produce training and validation data

    Attributes:
    -----------
        datasetpath ([str]): path to the folder which contains entire dataset/images
        idx_to_class (dict[value,empname]): dict mapping empnames to integer values
        stratify ([bool]): if to keep the positive and negative samples balanced in train and val set
        val_split ([float]): % of samples to put into val set. Default 10% of data
        shuffle ([bool]): If to shuffle the data samples before splitting

    """

    def __init__(self, datasetpath, stratify=False, val_split=0.1, shuffle=True):
        self.datasetpath = datasetpath

        self.class_to_idx = get_cls_to_idx(self.datasetpath)
        self.idx_to_class = get_idx_to_cls(self.class_to_idx)
        self.data, self.label, self.emp_names = self.create_data_label()

        self.stratify = stratify
        self.val_split = val_split
        self.shuffle = shuffle

    def create_data_label(self):
        """Prepares the image path, labels and empnames

        Returns:
            data,label,emp_names: data holds the path to the images, label is the integer value assigned to
                                  emps and emp_names is the name of the empeor
        """
        i = 0
        data = []
        label = []
        emp_names = []
        for foldernames in os.listdir(self.datasetpath):
            for files in os.listdir(os.path.join(self.datasetpath, foldernames)):
                if files == "desktop.ini":
                    print(f"[INFO] FILE MUST NOT BE HERE..")
                    exit
                data.append(os.path.join(self.datasetpath, foldernames, files))
                emp_names.append(foldernames)
                label.append(self.class_to_idx[foldernames])
        data = np.asarray(data)
        label = np.asarray(label)
        emp_names = np.asarray(emp_names)

        return data, label, emp_names

    def split(self):
        """Method to split the data based on the stratify flag and saves the train and split data
        in a excel file
        """
        if self.stratify:
            Xtrain, Xval, ytrain, yval = train_test_split(
                self.data,
                self.label,
                shuffle=self.shuffle,
                stratify=self.label,
                test_size=self.val_split,
                random_state=42,
            )
        else:
            Xtrain, Xval, ytrain, yval = train_test_split(
                self.data,
                self.label,
                shuffle=self.shuffle,
                test_size=self.val_split,
                random_state=42,
            )
        print(f"[INFO] Training dataset has - {len(Xtrain)} images")
        print(f"[INFO] Validation dataset has - {len(Xval)} images")

        print(f"[INFO] Creating training and validation dictionaries..")
        train_dict = create_dict(Xtrain, ytrain)
        val_dict = create_dict(Xval, yval)
        print(f"[INFO] DONE...")

        return train_dict, val_dict

        # self.save_excel(train_dict ,excel_name='train')

        # self.save_excel(val_dict,excel_name='valid')
        # print(f'[INFO] Writing the validation dictionaries to disk as excel file in - {cfg.VALID_EXCEL_PATH}')

        # print(f'[INFO] DONE.....')

    def save_excel(self, dictionary, excel_name="default"):
        """Saves the data in the dictionary into an excel sheet

        Args:
            dictionary (dict[imagepath,labels]): dictionary containing key as the image path and values as the integer labels
            excel_name (str, optional): Name of the excel sheet to be generated. Defaults to 'default'.
        """

        df = pd.DataFrame.from_dict(
            data={
                "ImagePath": [i for i in dictionary.keys()],
                "Label": [i for i in dictionary.values()],
            },
            orient="columns",
        )
        df["EmpName"] = pd.Series(
            data=[self.idx_to_class[i] for i in df.Label])

        # TODO:
        # add assertion check if ImagePath contains the EmpName

        if excel_name == "train":
            writer = pd.ExcelWriter(cfg.TRAIN_EXCEL_PATH)
        else:
            writer = pd.ExcelWriter(cfg.VALID_EXCEL_PATH)

        df.to_excel(writer, sheet_name=str(excel_name), index=False)
        writer.save()


if __name__ == "__main__":
    gen = TrainTest_Generator(
        datasetpath=cfg.DATASET_PATH, stratify=True, val_split=0.20
    )

    train_dict, val_dict = gen.split()

    print(
        f"[INFO] Writing the {excel_name} dictionaries to disk as excel file in - {cfg.TRAIN_EXCEL_PATH}",
        end=" ",
    )
    gen.save_excel(train_dict, "train")
    print("...OK!")

    print(
        f"[INFO] Writing the {excel_name} dictionaries to disk as excel file in - {cfg.VALID_EXCEL_PATH}",
        end=" ",
    )
    gen.save_excel(val_dict, "valid")
    print("...OK!")
