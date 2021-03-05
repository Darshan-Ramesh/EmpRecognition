import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from dict_mappings import get_cls_to_idx,get_idx_to_cls
from traintest_generator import TrainTest_Generator
import config as cfg
from sklearn.metrics import classification_report, confusion_matrix



# def get_df(excel_path):   
    
#     """Function to read the dataframe afer reading an excel file
    
#     Arguments
#     ---------
#     excel_path: str
#         Path of the excel sheet to read from
#     """
    
#     return pd.read_excel(excel_path)

def visualize_df(excel_path):
    """
    Function that plots a bar plot from the read dataframe. The dataframe must have 'EmpName' 
    column mentioning the names of the emperors
    
    Arguments
    ---------
    excel_path: str
        Path of the excel sheet to read from
    """
    
    df = pd.read_excel(excel_path)
    df.EmpName.value_counts(ascending=True).plot(kind='barh',figsize=(7,7))
    count = df.EmpName.value_counts(ascending=True)
    for index, value in enumerate(count):
        plt.text(value, index, str(value))
    
    plt.xlabel('Counts')
    plt.ylabel('Emperor Names')
    plt.title('Emp Names vs Counts')
    plt.tight_layout()
    plt.show()
    
    
def draw_confusion_mat(gt,y_hat,labels,normalized=None,figsize=(5,5)):
    """Plots a confusion matrix using the seaborn heatmpas

    Args:
        gt ([list]): Ground Truth Labels
        y_hat ([list]): Predictions from the model
        labels (list[str]): list of labels that are names mapped to corresponding "gt"
        normalized ([bool]): If to normalize and display the values. Defaults to None.
        figsize (size, size): Size of the figure to be displayed. Defaults to (5,5).
    """    
    
    # sns.set(font_scale=1.2)
    
    cf_mat = confusion_matrix(y_hat,gt,labels=labels,normalize=normalized)
    fig,ax = plt.subplots(figsize = figsize)
    
    res = sns.heatmap(cf_mat,annot=True,annot_kws={"size": 14},
                xticklabels=labels, yticklabels=labels,
                linewidth=0,
                cmap='Blues',
                square=True)
    
    
    res.set_yticklabels(res.get_ymajorticklabels(),fontsize = 14)
    res.set_xticklabels(res.get_xmajorticklabels(),fontsize = 14)
    
    
    
    title = cfg.LOAD_CHECKPOINT.split('\\')[-1]
    plt.xlabel("Predicted labels",loc='center',fontsize=14)
    plt.ylabel("Actual labels",loc='center',fontsize=14)
    # plt.title(title)
    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == "__main__":
    excel_path = cfg.TRAIN_EXCEL_PATH
    visualize_df(excel_path)