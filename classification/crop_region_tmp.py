from PIL import Image
import json
import argparse
import re
import os
import openslide
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

from classification_model import *

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural Network")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning  rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for trainging')
    parser.add_argument('--labels', type=int, default=2,
                        help="ResNet18 number of Class")
    parser.add_argument("--batch_size", type=int, default=32,
                        help='Model Batch_size')
    parser.add_argument('--model_save_path', type=str, default=f'Model_save/20231205_model1',
                        help="Where to Save model ")
    parser.add_argument('--pretrained', type=str,  default=False,
                        help="Using Pretrained or not")
    parser.add_argument('--loss_image_save', type=bool, default =False,
                        help="Sampling Image getting High Loss ")
    args = parser.parse_args()

    return args

args = get_args()

model_path = "/home/lab/Tumor_Detection/Model_save/20231228_model1/1_1/1_1/epoch_511_all.tar"
stride = 32 * 4


# base_path : CLAM의 결과 (png, h5 파일이 있는 directory)
base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"
# svs_base_path : svs 파일이 있는 directory
svs_base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/demo/slides/"
# patch_base_path : pacth가 있는 directory
patch_base_path = "/home/lab/Tumor_Detection/classification/224_patch_score_80"
# heatmap_base_path : heatmap이 있는 directory
heatmap_base_path = "/home/lab/Tumor_Detection/classification/heatmap"
# output_path : 저장할 경로
output_path = "/home/lab/Tumor_Detection/classification/cropped_region"
if not os.path.exists(output_path):
    os.makedirs(output_path)


checkpoint = torch.load(model_path)
model_state_dict = checkpoint['model']

model = make_MobileNetV2(args)
model.load_state_dict(model_state_dict)

device = torch.device("cuda:0")
model.to(device)
model.eval()


# for filename in os.listdir(heatmap_folder):
#     if filename.endswith(".png"):
#         pattern = r"_(\d+)_(\d).png"
#         matches = re.findall(pattern, filename)
#         print(matches)
#         # if matches:
            


def crop_region(tumor_name, pos_or_neg, output_path, stride):
    model.eval()
    
    sigmoid = nn.Sigmoid()

    output_path = os.path.join(output_path, pos_or_neg +  "_" + tumor_name)
    patch_path = os.path.join(patch_base_path, tumor_name)

    CLAM_image = Image.open(os.path.join(base_path, tumor_name, f"{tumor_name}_mask.jpg"))
    heatmap_image = Image.open(os.path.join(heatmap_base_path, pos_or_neg + "_" + tumor_name + ".jpg"))

    svs_path = os.path.join(svs_base_path, tumor_name + ".svs")
    svs_image = openslide.open_slide(svs_path)

    width, height = CLAM_image.size

    width = int(width)
    height = int(height)

    coordinates_list = extract_coord(patch_path)
    print(len(coordinates_list))


    with torch.no_grad():
        valid_coordinates = []
        for w in tqdm(range(0, width-96, 32)):
            for h in range(0, height-96, 32):
                wrong = 0
                for coordinate in coordinates_list:
                    x, y = coordinate
                    if w <= x <= w + stride and h <= y <= h + stride:
                        valid_coordinates.append(coordinate)
                    # valid_coordinates안에는 window안에 있는 patch개수가 있을거임
                    wrong += prob_calc(valid_coordinates, pos_or_neg, patch_base_path, tumor_name)
                    total = len(valid_coordinates)
                    if total != 0:
                        print(total, wrong)
                    

    
def extract_coord(folder_path):
    coordinates_list = []
    for patch in os.listdir(folder_path):
        if patch.endswith(".png"):
            pattern = r"_(\d+)_(\d+)\.png"
            matches = re.findall(pattern, patch)

            for match in matches:
                coords = [int(value) // 8 for value in match]
                coordinates_list.append(coords)
    return coordinates_list


def prob_calc(coord_list, pos_or_neg, patch_path, tumor_name):
    wrong = 0

    model.eval()
    transform = transforms.ToTensor()
    sigmoid = nn.Sigmoid()
    for coordinate in coord_list:
        x, y = coordinate
        x = x * 8
        y = y * 8
        patch = Image.open(os.path.join(patch_path, tumor_name,  f"{tumor_name}_{x}_{y}.png"))
        image = transform(patch).to(device).unsqueeze(dim=0)

        output = model(image)
        probs_neg = sigmoid(output)[0][0].item()
        probs_pos = sigmoid(output)[0][0].item()

        if pos_or_neg == "pos":
            if probs_neg >= 0.6:
                wrong += 1
            else:
                wrong += 0
        else:
            if probs_pos >= 0.6:
                wrong += 1
            else:
                wrong += 0
    return wrong





crop_region("SS22-26507_55000_23000", "pos", output_path, stride)