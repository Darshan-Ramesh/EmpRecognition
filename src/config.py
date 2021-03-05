import torch
import pandas as pd

__name__ = 'config'

SEED = 1337
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 40

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8

##################################################################
###################### for face recogniotion #####################
##################################################################
#train_new.xlsx = removed non-emperor photos
#train.xlsx = including non-emperor photos 
DATASET_PATH = "..\\Datasets\\v5\\keizer_v5\\onlyfaces\\bynames\\"
TRAIN_EXCEL_PATH =  "..\\input\\v5\\aligned\\faces_256x256\\train.xlsx"
VALID_EXCEL_PATH =  "..\\input\\v5\\aligned\\faces_256x256\\valid.xlsx"
TEST_EXCEL_PATH =  "..\\input\\v5\\aligned\\faces_256x256\\test.xlsx"

#31 for face and #30 for hair
CLASSES = 10

##################################################################
###################### for hair classification ###################
##################################################################
# DATASET_PATH = "..\\Datasets\\v5\\keizer_v5\\onlyhair\\bynames\\"
# TRAIN_EXCEL_PATH =  "..\\input\\v5\\onlyhair\\train.xlsx"
# VALID_EXCEL_PATH =  "..\\input\\v5\\onlyhair\\valid.xlsx"


COLUMNS = ['ImagePath','Label','EmpName']

OPTIMIZER = 'ADAMW'   #['ADAM','ADAMW','SGD']
STEPS = [5,30,75]
# W_DECAY= 0.01
W_DECAY = 0
LR_RATE = 0.005

BETA = 0.999
GAMMA = 2.0                 #[for class balanced loss it is 0.5 from their paper]
LOSS_FN = 'ce'              #['ce','focal','cb_focal']

#model name
#['Inception', 'resnet50','senet50']
MODEL = 'Inception'              

#['ArcFace, 'CirriculumFace']
HEAD = None        
     
CLASSIFY = True

#augmentations
AUGMENTATION = True

#sampling technique
WEIGHTED_SAMPLING = True

#saving models and excel sheet of misclassifications
SAVE_MODEL = True
SAVE_HEAD = False

if not CLASSIFY:
    SAVE_HEAD = True


MODEL_BASE_FOLDER = "..\\models\\"
LOGS_BASE_FOLDER = "..\\logs\\prediction_logs\\"

CHECKPOINT_NAME = "Exp1_test.pt"
CHECKPOINT_NAME_HEAD = "Head_" + "Inception_Aug_7.pt"
PATH = "v5\\aligned\\face\\Inception\\"

#MODEL SAVE
SAVE_CHECKPOINT = MODEL_BASE_FOLDER + PATH + CHECKPOINT_NAME
#HEAD SAVE
SAVE_CHECKPOINT_HEAD = MODEL_BASE_FOLDER + PATH + CHECKPOINT_NAME_HEAD


#prediction related
LOAD_CHECKPOINT =  MODEL_BASE_FOLDER + PATH + CHECKPOINT_NAME
LOAD_HEAD_CHECKPOINT = MODEL_BASE_FOLDER + PATH + CHECKPOINT_NAME_HEAD

PREDICTION_LOGS = LOGS_BASE_FOLDER + PATH
RUNS = PREDICTION_LOGS + "\\Exp2_test\\"

WRITE_TO_EXCEL = False
EXCEL_NAME = 'Exp2_test.xlsx'

#copy this config file and rename to a .txt file
CONFIG_FILE_PATH = "config.py"

RESUME = False 
if RESUME:
    PREV_CHECKPOINT_NAME = "1_1.pt"
    PREV_LOAD_CHECKPOINT =  MODEL_BASE_FOLDER + PATH + PREV_CHECKPOINT_NAME
    
    # PREV_CHECKPOINT_NAME_HEAD = "Head_" + "Inception_Aug_7_Linearpythn.pt"
    # PREV_LOAD_HEAD_CHECKPOINT = MODEL_BASE_FOLDER + PATH + PREV_CHECKPOINT_NAME_HEAD