from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torch.nn as nn
from PIL import Image, ImageDraw
import re
import torchvision.transforms as transforms
import json
from tqdm import tqdm
import time

base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"

def find_CLAM_JPEG(folder_name, tumor_name) :
    matching_name_length = len(tumor_name)

    for image_name in os.listdir(CLAM_PATH):
        if tumor_name == image_name[:matching_name_length]:
            return Image.open(os.path.join(CLAM_PATH, image_name))
        
def make_HeatMap(model, device):
    model.eval()
    transform = transforms.ToTensor()
    sigmoid = nn.Sigmoid()

    TEST_PATH = []
    with open('/home/lab/Tumor_Detection/tumorAnnotation.json', 'r') as f:
        json_data = json.load(f)


    for k,v in json_data.items():
        if k == 'NEGATIVE_TEST_LUAC':
            TEST_PATH.extend(v)

    # print(f"test path : {TEST_PATH}")

    for path in tqdm(base_path):
        folder_name, tumor_name = path.split('/')[0:1], path.split('/')[2]

        print(f'folder_name : {folder_name} |  tumor_name : {tumor_name}')

        CLAM_RESULT_JPEG = find_CLAM_JPEG(folder_name, tumor_name)
        print(CLAM_RESULT_JPEG.filename)

        white_Image = Image.new('RGB', (CLAM_RESULT_JPEG.width, CLAM_RESULT_JPEG.height), (255, 255, 255))
        draw = ImageDraw.Draw(white_Image)


        patch_path = os.path.join(folder_name[0], "NEGATIVE_250", tumor_name)
        print(patch_path)
        patch_list = os.listdir(patch_path)

        with torch.no_grad():
            for patch in patch_list:
                # print(patch)
                pattern = r"_(\d+)_(\d+)\.png"
                matches = re.findall(pattern, patch)
                # print(matches)

                width, height = matches[0]
                width, height = int(width), int(height)

                patch = Image.open(os.path.join(patch_path, patch))
                image = transform(patch).to(device).unsqueeze(dim = 0)

                output = model(image)
                probs = sigmoid(output)[0][0].item()

                # print(f"Width : {width} | Height : {height}      | Prob : {probs}" )
                red, blue = int(probs * 255), int((1-probs) * 255)

                draw.rectangle((width, height, width+2, height+2), fill =(red, 0 ,blue))

        heatmap_folder = "heatmap"
        if not os.path.exists(heatmap_folder):
            os.makedirs(heatmap_folder)
        white_Image.save(os.path.join(heatmap_folder, tumor_name + '.jpg'))