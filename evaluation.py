import os
import numpy as np
import argparse
import cv2
import seaborn as sns
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from utils.ImageHelper import ImageHelper
from typing import AnyStr, Union, Tuple
from glob import glob
from utils import EvaluationHelper
from tqdm import tqdm
from PIL import Image
#from multiprocessing import Pool


#this code is for calculate assd and dc (default = .png and 2d)  ←change it as needed
#you have to set up output and prediction and label path for this running
# by yuruihito in 2024 

LABEL_NAME_DICT = {0: "Foreground", 1: "Background", 2: "Not-classified"}#change it as needed

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir",
                        type=str,
                        default="<your path>",
                        help="Work space directory for this running, ")
    parser.add_argument("--pred_dir",
                        type=str,
                        default="<your path>",
                        help="prediction directory for this running, ")
    parser.add_argument("--target_dir",
                        type=str,
                        default="<your path>",
                        help="label directory for this running, ")
    parser.add_argument("--image_dir",
                        type=str,
                        default="<your path>",
                        help="images directory for this running, ")
    parser.add_argument("--data_size",
                        type=Tuple[int ,int],
                        default=(224,224),
                        help="Work space directory for this running, ")

    opt = parser.parse_args()
    # set up  path 
    saving_size = opt.data_size
    eval_save_dir = os.path.join(opt.work_space_dir, "_eval")
    os.makedirs(eval_save_dir, exist_ok=True)
    print(eval_save_dir)    
    imgs = glob(os.path.join(opt.target_dir, "*.png"))

    try:
        image_ids = [os.path.splitext(os.path.basename(filename))[0] for filename in imgs]
    except:
        print("Path Error")
        print(imgs)

    #dc and assd
    data = []  
    for image_id in tqdm(image_ids,
                        desc = "calculation"):
        pred_path = os.path.join(opt.pred_dir, f"{image_id}_label.png")
        target_path = os.path.join(opt.target_dir, f"{image_id}.png")
        dc, assd =  calc_dc_assd(target_path, pred_path)
        mean_dc = 0
        mean_assd = 0
        for class_id in LABEL_NAME_DICT:
            data.append({"image_id": image_id, "class": LABEL_NAME_DICT[class_id], "DC": dc[class_id], "ASSD": assd[class_id]})
            mean_dc += dc[class_id]
            mean_assd += assd[class_id]
        mean_dc /= len(LABEL_NAME_DICT)
        mean_assd /= len(LABEL_NAME_DICT)
        data.append({"image_id": image_id, "class": "Mean", "DC": mean_dc, "ASSD": mean_assd})
    print("calculation done")

    #to excel
    print("data")
    print(data[0])  
    eval_df = pd.DataFrame(data, index=range(len(data)))
    eval_df.to_excel(os.path.join(eval_save_dir, "eval.xlsx"))   
    print("output DataFrame")

    _draw_boxplot(data=eval_df,
                  x="class",
                  y="DC",
                  save_path=os.path.join(eval_save_dir, "dc.png"))
    _draw_boxplot(data=eval_df,
                  x="class",
                  y="ASSD",
                  save_path=os.path.join(eval_save_dir, "assd.png"))
    print("output Boxplot")
    # print(f'dc {dc}')
    # print(f'assd {assd}')

    # Sample best, median, and worst
    eval_df = eval_df[eval_df["class"] == "Mean"]
    summary(eval_df,"DC",saving_size,opt.image_dir,opt.target_dir,opt.pred_dir,eval_save_dir)
    summary(eval_df,"ASSD",saving_size,opt.image_dir,opt.target_dir,opt.pred_dir,eval_save_dir)

#boxplot
def _draw_boxplot(data: pd.DataFrame,
                  x: AnyStr,
                  y: AnyStr,
                  save_path: AnyStr,
                  order=None,
                  palette="Blues",
                  figsize=None,
                  dpi=180):
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = sns.boxplot(data=data, x=x, y=y, order=order, palette=palette, dodge=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(ax.figure)
    plt.close(fig)
    plt.close()

#assd and dc
def calc_dc_assd(gt_path, pred_path):
    labels_array = cv2.imread(gt_path)
    pred_labels_array  = cv2.imread(pred_path)
    unique_labels = np.unique(labels_array)

    dice_coefficients = {}
    assd_val = {}

    for label in unique_labels:
        pred_label = np.where(pred_labels_array == label, 1, 0)
        label_mask = np.where(labels_array == label, 1, 0)

        dice_coefficients[label], assd_val[label] = Evaluation.EvaluationHelper.dc_and_assd(pred_label, label_mask)


    return dice_coefficients, assd_val

#color save
def __save_sample_visual(imgae_path: AnyStr,
                         target_label_path: AnyStr,
                         pred_label_path: AnyStr,
                         image_dsize: tuple[int, int],
                         save_dir: AnyStr,
                         name_prefix: AnyStr) -> None:
    image = Image.open(imgae_path).convert("RGB")
    image = image.resize(image_dsize, Image.BILINEAR)
    image.save(os.path.join(save_dir, f"{name_prefix}_Image.png"))

    min_class_id = min(LABEL_NAME_DICT.keys())
    max_class_id = max(LABEL_NAME_DICT.keys())

    target_label = np.array(Image.open(target_label_path)) 
    target_label = cv2.resize(target_label, image_dsize, interpolation=cv2.INTER_NEAREST)
    target_label = ImageHelper.apply_colormap_to_dense_map(target_label, min_class_id, max_class_id)
    cv2.imwrite(os.path.join(save_dir, f"{name_prefix}_True.png"), target_label)
    
    pred_label = np.array(Image.open(pred_label_path)) 
    pred_label = cv2.resize(pred_label, image_dsize, interpolation=cv2.INTER_NEAREST)
    pred_label = ImageHelper.apply_colormap_to_dense_map(pred_label, min_class_id, max_class_id)
    cv2.imwrite(os.path.join(save_dir, f"{name_prefix}_Pred.png"), pred_label)

def summary(eval_df,
            eval,
            saving_size,
            image_dir,
            target_dir,
            pred_dir,
            eval_save_dir):
    eval_df.sort_values(by=f"{eval}", ascending=False if eval == "DC" else True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    eval_df = eval_df.loc[[0, len(eval_df) // 2, len(eval_df) - 1]]
    for prefix, (_, row) in zip(["max", "median", "min"], eval_df.iterrows()):
        print(prefix)
        print(row)
        image_path = os.path.join(image_dir, f"{row['image_id']}_0000.png")
        target_label_path = os.path.join(target_dir, f"{row['image_id']}.png")
        pred_label_path = os.path.join(pred_dir, f"{row['image_id']}_label.png")
        _eval_save_dir = os.path.join(eval_save_dir,f"mean_{eval}")
        os.makedirs(_eval_save_dir, exist_ok=True)      
        __save_sample_visual(image_path,target_label_path, pred_label_path,saving_size,_eval_save_dir,prefix)

if __name__ == '__main__':
    main()