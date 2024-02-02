import torch
import torchvision
import torch.nn as nn
from PIL import Image, ImageDraw
import re
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import os
import argparse

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

# base_path : CLAM의 결과 (png, h5 파일이 있는 directory)
base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"
# svs_base_path : svs 파일이 있는 directory
svs_base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/demo/slides/"

patch_base_path = "/home/lab/Tumor_Detection/classification/224_patch_score_80"

model_path = "/home/lab/Tumor_Detection/Model_save/20231228_model1/1_1/1_1/epoch_511_all.tar"
checkpoint = torch.load(model_path)
model_state_dict = checkpoint['model']

model = make_MobileNetV2(args)
model.load_state_dict(model_state_dict)

device = torch.device("cuda:0")
model.to(device)
model.eval()

def heatmap(tumor_name, pos_or_neg):

    model.eval()
    transform = transforms.ToTensor()
    sigmoid = nn.Sigmoid()

    CLAM_image = Image.open(os.path.join(base_path, tumor_name, f"{tumor_name}_mask.jpg"))
    # white_image = Image.new('RGB', (CLAM_image.width, CLAM_image.height), (255, 255, 255))
    draw_clam = ImageDraw.Draw(CLAM_image, "RGBA")
    # draw_white = ImageDraw.Draw(white_image)

    patch_path = os.path.join(patch_base_path, tumor_name)
    print(patch_path)

    with torch.no_grad():
        for patch in os.listdir(patch_path):
            if patch.endswith(".png"):
                print(patch)
                pattern = r"_(\d+)_(\d+)\.png"
                matches = re.findall(pattern, patch)
                # print(matches)

                width, height = matches[0]
                width, height = int(width)/8, int(height)/8
                # print(width, height)

                patch = Image.open(os.path.join(patch_path, patch))
                image = transform(patch).to(device).unsqueeze(dim=0)

                output = model(image)
                probs = sigmoid(output)[0][0].item()
                probs_tmp = sigmoid(output)[0][1].item()
                print(probs, probs_tmp)

                # print(f"width : {width} |   height : {height}       |   prob : {probs}")
                red, blue = int(probs * 255), int(probs_tmp * 255)
                draw_clam.rectangle((width, height, width + 32, height + 32), fill=(red, 0, blue, 125))
                # draw_white.rectangle((width, height, width + 32, height + 32), fill=(red, 0, blue))

    heatmap_folder = "heatmap"

    if not os.path.exists(heatmap_folder):
        os.makedirs(heatmap_folder)

    CLAM_image.save(os.path.join(heatmap_folder, pos_or_neg + "_" + tumor_name + ".jpg"))
    # white_image.save(os.path.join(heatmap_folder, tumor_name + "_white.jpg"))


with open("../tumorAnnotation.json", "r") as file:
    data = json.load(file)


keys_pos = ["POSITIVE_TEST_YS", "POSITIVE_TEST_TCGA", "POSITIVE_TEST_LUAC", "POSITIVE_TEST_SSSF"]
keys_neg = ["NEGATIVE_TEST_LUAC", "NEGATIVE_TEST_SSSF"]

for key in keys_pos:
    current_list = data.get(key, [])
    for value in current_list:
        value = value.split('/')[-1]
        heatmap(value, "pos")


for key in keys_neg:
    current_list = data.get(key, [])
    for value in current_list:
        value = value.split('/')[-1]
        heatmap(value, "neg")