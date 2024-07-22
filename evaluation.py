import os
import re 
import numpy as np
import argparse
import cv2
import seaborn as sns
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import SimpleITK as sitk
import shutil
from IPython.display import display
from utils.ImageHelper import ImageHelper
from typing import AnyStr, Union, Tuple
from glob import glob
from utils import Evaluation
from tqdm import tqdm
from PIL import Image
#from multiprocessing import Pool


#this code is for calculate assd and dc, asd (default = .png and 2d)  
#you have to set up output and prediction and label path for this running 

#oseteonecrosis
LABEL_NAME_DICT =  {0: "label0", 1: "label1", 2: "label2", 3: "label3"}
#LABEL_NAME_DICT =  {0: "Foreground", 1: "Background", 2: "Not-classified"}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir",
                        type=str,
                        default="//your path",
                        help="Work space directory for this running, ")
    parser.add_argument("--pred_dir",
                        type=str,
                        default="//your path",
                        help="prediction directory for this running, ")
    parser.add_argument("--target_dir",
                        type=str,
                        default= "//your path",
                        help="label directory for this running, ")
    parser.add_argument("--image_dir",
                        type=str,
                        default="//your path",
                        help="images directory for this running, ")
    parser.add_argument("--data_size",
                        type=Tuple[int ,int],
                        default=(224,224),
                        help="Work space directory for this running, ")
    parser.add_argument("--fold",
                        type = str,
                        default = 0)
    parser.add_argument("--colormaps",
                        type = bool,
                        default = False)
    opt = parser.parse_args()
    # set up  path 
    saving_size = opt.data_size
    eval_save_dir = os.path.join(opt.work_space_dir, f"osteonecrosis_eval_fold{opt.fold}")
    os.makedirs(eval_save_dir, exist_ok=True)
    print(f"evaluation_savedir:{eval_save_dir}")    
    #imgs = glob(os.path.join(opt.target_dir, "*.png"))
    imgs = glob(os.path.join(opt.target_dir, "*.nii.gz"))
    print(imgs)
                
    try:
        #image_ids = [os.path.splitext(os.path.basename(filename))[0] for filename in imgs]
        image_ids = [(re.match(r'(.+)\.nii\.gz$', os.path.basename(filename))).group(1) for filename in imgs]
    except:
        print("Path Error")
        
    print(LABEL_NAME_DICT)
    #dc asd assd
    data = []
    print("assd_asd_dc")
    for image_id in tqdm(image_ids,
                        desc = "calculation"):
        #pred_path = os.path.join(opt.pred_dir, f"{image_id}_label.png")
        pred_path = os.path.join(opt.pred_dir, f"{image_id}_label.nii.gz")
        #target_path = os.path.join(opt.target_dir, f"{image_id}.png")
        target_path = os.path.join(opt.target_dir, f"{image_id}.nii.gz")
        dc, assd, asd=  calc_dc_assd_asd(target_path, pred_path)
        mean_dc = 0
        mean_assd = 0
        mean_asd = 0
        for class_id in LABEL_NAME_DICT:
            data.append({"image_id": image_id, "class": LABEL_NAME_DICT[class_id], "DC": dc[class_id], "ASSD": assd[class_id], "ASD": asd[class_id]})
            mean_dc += dc[class_id]
            mean_assd += assd[class_id]
            mean_asd = asd[class_id]
        mean_dc /= len(LABEL_NAME_DICT)
        mean_assd /= len(LABEL_NAME_DICT)
        mean_asd /= len(LABEL_NAME_DICT)
        data.append({"image_id": image_id, "class": "Mean", "DC": mean_dc, "ASSD": mean_assd, "ASD": mean_asd})
    print("calculation done")
      
    #to excel
    print("data")
    print(data[0])
    print()
    EVAL_LIST = ["DC","ASD","ASSD"]  
    eval_df = pd.DataFrame(data, index=range(len(data)))
    display(eval_df)
    print("avg")
    group = eval_df.groupby("class", sort = False)
    df = group.aggregate({"DC":"mean","ASSD":"mean","ASD":"mean"})
    df = df.T
    display(df)
    print("std")
    group_ = eval_df.groupby("class", sort = False)
    df_ = group_.aggregate({"DC":"std","ASSD":"std","ASD":"std"})
    df_ = df_.T
    display(df_)
    
    dc = maek_dataframe(df, df_, EVAL_LIST[0])
    asd = maek_dataframe(df, df_, EVAL_LIST[1])
    assd = maek_dataframe(df, df_, EVAL_LIST[2])

    #dataframe for plot
    pivot_df_dc_ = eval_df.pivot(index="class", columns="image_id", values=["DC"])
    pivot_df_dc = pivot_df_dc_.T
    pivot_df_asd_ = eval_df.pivot(index="class", columns="image_id", values=["ASD"])
    pivot_df_asd = pivot_df_asd_.T
    pivot_df_assd_ = eval_df.pivot(index="class", columns="image_id", values=["ASSD"])
    pivot_df_assd = pivot_df_assd_.T

    #boxplot
    eval_df.to_excel(os.path.join(eval_save_dir, "eval.xlsx"))   
    print("output DataFrame")

    _draw_boxplot(data=eval_df,
                  eval=dc,
                  eval_=pivot_df_dc,
                  x="class",
                  y="DC",
                  save_path=os.path.join(eval_save_dir, "dc.png"))
    _draw_boxplot(data=eval_df,
                  eval=asd,
                  eval_=pivot_df_assd,
                  x="class",
                  y="ASSD",
                  save_path=os.path.join(eval_save_dir, "assd.png"))
    _draw_boxplot(data=eval_df,
                  eval=assd,
                  eval_=pivot_df_asd,
                  x="class",
                  y="ASD",
                  save_path=os.path.join(eval_save_dir, "asd.png"))
    print("output Boxplot")
    # print(f'dc {dc}')
    # print(f'assd {assd}')

    # Sample best, median, and worst
    eval_df = eval_df[eval_df["class"] == "Mean"]
    summary(eval_df,"DC",saving_size,opt.image_dir,opt.target_dir,opt.pred_dir,eval_save_dir,opt.colormaps)
    summary(eval_df,"ASD",saving_size,opt.image_dir,opt.target_dir,opt.pred_dir,eval_save_dir,opt.colormaps)
    summary(eval_df,"ASSD",saving_size,opt.image_dir,opt.target_dir,opt.pred_dir,eval_save_dir,opt.colormaps)

def maek_dataframe(df, df_, eval):
        print(eval)
        val = pd.concat([df.loc[eval],df_.loc[eval]], axis = 1)
        val = val.T
        val.index = ["Avg","Std"]
        display(val)
        return val

#boxplot
def _draw_boxplot(data: pd.DataFrame,
                  eval: pd.DataFrame,
                  eval_: pd.DataFrame,
                  x: AnyStr,
                  y: AnyStr,
                  save_path: AnyStr,           
                  order=None,
                  palette="Oranges",
                  figsize=None,
                  dpi=180):
    plt.clf()
    # fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=data, x=x, y=y, order=order, palette=palette, dodge=False, ax=ax)
    ax_ = sns.stripplot(data=eval_, palette='dark:green', jitter=True, alpha=0.5, ax=ax)
    if "SD" in y:
        if y == "ASSD":
            plt.ylabel("ASSD[mm]")
        else:
            plt.ylabel("ASD[mm]")

    ax_table = plt.subplot(111, frame_on=False)#L,M,N
    ax_table.axis('off')  # 枠線を非表示
    table = ax_table.table(cellText=eval.round(4).values,
                   colLabels=None,
                   rowLabels=eval.index,
                   loc='bottom',
                   cellLoc='center',
                   rowColours=['lightblue']*len(eval.index),
                   bbox=[0, -0.4, 1, 0.25])
    
    table.auto_set_font_size(False)  # 自動のフォントサイズ設定をオフ
    table.set_fontsize(9)#fontsize

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(ax.figure)
    plt.close(ax_.figure)
    plt.close(fig)
    plt.close()

#assd and dc
def calc_dc_assd_asd(gt_path, pred_path):
    #labels_array = cv2.imread(gt_path)
    #pred_labels_array  = cv2.imread(pred_path)
    label_img = sitk.ReadImage(gt_path)
    pred_labels_img  = sitk.ReadImage(pred_path)

    labels_array = sitk.GetArrayFromImage(label_img)
    pred_labels_array  = sitk.GetArrayFromImage(pred_labels_img)
    unique_labels = np.unique(labels_array)

    dice_coefficients = {}
    assd_val = {}
    asd_val = {}

    for label in unique_labels:
        pred_label = np.where(pred_labels_array == label, 1, 0)#np.where(pred_labels_array == label, 1, 0):labelであれば1そうでなければ0を返す
        label_mask = np.where(labels_array == label, 1, 0)

        dice_coefficients[label], assd_val[label] = Evaluation.EvaluationHelper.dc_and_assd(pred_label, label_mask)
        asd_val[label] = Evaluation.EvaluationHelper.asd(pred_label, label_mask)


    return dice_coefficients, assd_val, asd_val

#color save
def __save_sample_visual(image_path: AnyStr,
                         target_label_path: AnyStr,
                         pred_label_path: AnyStr,
                         image_dsize: tuple[int, int],
                         save_dir: AnyStr,
                         name_prefix: AnyStr,
                         colormaps) -> None:
    if colormaps:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(image_dsize, Image.BILINEAR)
        image.save(os.path.join(save_dir, f"{name_prefix}_Image.png"))

        min_class_id = min(LABEL_NAME_DICT.keys())
        max_class_id = max(LABEL_NAME_DICT.keys())

        target_label = np.array(Image.open(target_label_path)) 
        target_label = cv2.resize(target_label, image_dsize, interpolation=cv2.INTER_NEAREST)
        target_label = ImageHelper.apply_colormap_to_dense_map(target_label, min_class_id, max_class_id)#カラーマップ
        cv2.imwrite(os.path.join(save_dir, f"{name_prefix}_True.png"), target_label)
        
        pred_label = np.array(Image.open(pred_label_path)) 
        pred_label = cv2.resize(pred_label, image_dsize, interpolation=cv2.INTER_NEAREST)
        pred_label = ImageHelper.apply_colormap_to_dense_map(pred_label, min_class_id, max_class_id)
        cv2.imwrite(os.path.join(save_dir, f"{name_prefix}_Pred.png"), pred_label)
    else :
        image = os.path.basename(image_path)
        output_path = os.path.join(save_dir,image)
        shutil.copy(image_path, output_path)

        target = os.path.basename(target_label_path)
        output_path = os.path.join(save_dir,target)
        shutil.copy(target_label_path, output_path)

        pred = os.path.basename(pred_label_path)
        output_path = os.path.join(save_dir,pred)
        shutil.copy(pred_label_path, output_path)

def summary(eval_df,
            eval,
            saving_size,
            image_dir,
            target_dir,
            pred_dir,
            eval_save_dir,
            colormaps):
    eval_df.sort_values(by=f"{eval}", ascending=False if eval == "DC" else True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    eval_df = eval_df.loc[[0, len(eval_df) // 2, len(eval_df) - 1]]
    for prefix, (_, row) in zip(["max", "median", "min"], eval_df.iterrows()):#zip(複数のリストを同時に取得),iterrows():(index,series)を一行ずつ取得
        #image_path = os.path.join(image_dir, f"{row['image_id']}_0000.png")nii.gz
        image_path = os.path.join(image_dir, f"{row['image_id']}_0000.nii.gz")#WIP
        #target_label_path = os.path.join(target_dir, f"{row['image_id']}.png")
        target_label_path = os.path.join(target_dir, f"{row['image_id']}.nii.gz")#WIP
        #pred_label_path = os.path.join(pred_dir, f"{row['image_id']}_label.png")
        pred_label_path = os.path.join(pred_dir, f"{row['image_id']}_label.nii.gz")#WIP

        eval_save_dir_ = os.path.join(eval_save_dir,f"mean_{eval}")
        os.makedirs(eval_save_dir_, exist_ok=True)      
        __save_sample_visual(image_path,target_label_path, pred_label_path,saving_size,eval_save_dir_,prefix,colormaps)

if __name__ == '__main__':
    main()