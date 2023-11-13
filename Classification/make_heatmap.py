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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = "./heatmap"

CLAM_jpeg = Image.open(jpeg_list[0]).convert('RGB')

print(CLAM_jpeg.width)

def make_Heatmap(CLAM_RESULT_JPEG, device):
    white_Image = Image.new('RGB', (CLAM_RESULT_JPEG.width, CLAM_RESULT_JPEG.height), (255, 255, 255))
    draw = ImageDraw.Draw(white_Image)

    with torch.no_grad():
        for patch in os.listdir(path):
            pattern = r"\#(\d+)_(\d+)\.png"
            matches = re.findall(pattern, patch)

            width, height = matches[0]
            width, height = int(width), int(height)

            patch = Image.open(os.path.join(path, patch))
            image = transform(patch).to(device).unsqueeze(dim = 0)

            output = model(image)
            probs = sigmouid(output)[0][0].item()

            print(f"Width : {width} | Height : {height}      | Prob : {probs}" )
            red, blue = int(probs * 255), int((1-probs) * 255)

            draw.rectangle((width, height, width+2, height+2), fill =(red, 0 ,blue))

        white_Image.save("LUCA_7-1_E6_Heatmap.jpg")

make_Heatmap(CLAM_jpeg, device)