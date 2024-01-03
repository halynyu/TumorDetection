from PIL import Image, ImageDraw, ImageCms
import h5py
from openslide import OpenSlide
import os
from tqdm import tqdm
import time
import numpy as np
import json
import pandas as pd
import cv2

# base_path : CLAM의 결과 (png, h5 파일이 있는 directory)
base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"
# svs_base_path : svs 파일이 있는 directory
svs_base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/demo/slides/"

def patching(slide, y, x, PATCH_SIZE, tau):
    level = 0
    image = slide.read_region((y * tau, x * tau,), level, (PATCH_SIZE, PATCH_SIZE))
    patch = np.array(image.convert('RGB'))
    return patch

def make_patch(json_file, patch_size, score_threshold):

    heatmap_path = f"/home/lab/Tumor_Detection/classification/heatmap_{score_threshold}"
    if not os.path.exists(heatmap_path):
        os.makedirs(heatmap_path)

    with open(json_file, "r") as file:
        data = json.load(file)
    
    for category, values in data.items():
        for value in tqdm(values):
            svs_name = value.split("/")[-1]
            if not os.path.exists(f"patch_score_{score_threshold}/{svs_name}"):
                os.makedirs(f"patch_score_{score_threshold}/{svs_name}")

            output_path =  f"patch_score_{score_threshold}/{svs_name}"
            svs_file = OpenSlide(os.path.join(svs_base_path + svs_name + ".svs"))
            blockmap_file = cv2.imread(os.path.join(base_path, svs_name, svs_name + "_blockmap.png"))
            csv_file = os.path.join(base_path, svs_name, svs_name + "_all_patches_scores.csv")

            df = pd.read_csv(csv_file)
            filtered_df = df[df["score"] >= score_threshold]
            for index, row in filtered_df.iterrows():
                x, y = int(row["coord_x"]), int(row["coord_y"])
                
                region = svs_file.read_region((x, y), 0, (256, 256))

                PIL_image = Image.new("RGB", region.size)
                PIL_image.paste(region, (0, 0))
                
                PIL_output_path = os.path.join(output_path, f"{svs_name}_{x}_{y}.png")

                PIL_image.save(PIL_output_path)
                # svs_file.close()
                
                x, y = int(x/8), int(y/8)
                top_left = (x, y)
                bottom_right = (x+32, y+32)
                cv2.rectangle(blockmap_file, top_left, bottom_right, (0, 0, 0), 3)
                
            cv2.imwrite(os.path.join(heatmap_path, f"{svs_name}_{score_threshold}.png"), blockmap_file)


patch_size=32
score_threshold = 80

json_file = "/home/lab/Tumor_Detection/tumorAnnotation.json"
make_patch(json_file, patch_size, score_threshold)