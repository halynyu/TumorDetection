from PIL import Image
import json
import argparse
import re
import os
import openslide

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




window_size = 128
threshold_percent = 60

# base_path : CLAM의 결과 (png, h5 파일이 있는 directory)
base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"
# svs_base_path : svs 파일이 있는 directory
svs_base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/demo/slides/"
# heatmap_base_path : 생성되는 heatmap 파일이 있는 directory
heatmap_base_path = "/home/lab/Tumor_Detection/classification/heatmap"
output_path = "/home/lab/Tumor_Detection/classification/cropped_region"
patch_base_path = "/home/lab/Tumor_Detection/classification/224_patch_score_80"

model_path = "/home/lab/Tumor_Detection/Model_save/20231228_model1/1_1/1_1/epoch_511_all.tar"
checkpoint = torch.load(model_path)
model_state_dict = checkpoint['model']

model = make_MobileNetV2(args)
model.load_state_dict(model_state_dict)

device = torch.device("cuda:0")
model.to(device)
model.eval()



def crop_region(tumor_name, pos_or_neg, output_path):
    # 사진 하나 찝어서 가져왔다고 가정
    
    model.eval()
    transform = transforms.ToTensor()
    sigmoid = nn.Sigmoid()

    output_path = os.path.join(output_path, pos_or_neg + "_" + tumor_name)
    patch_path = os.path.join(patch_base_path, tumor_name)
    
    # CLAM 이미지 가져오기
    CLAM_image = Image.open(os.path.join(base_path, tumor_name, f"{tumor_name}_mask.jpg"))
    # heatmap 이미지 가져오기
    heatmap_image = Image.open(os.path.join(heatmap_base_path, pos_or_neg + "_" + tumor_name + ".jpg"))
    # 원본 svs이미지 가져오기
    svs_path = os.path.join(svs_base_path, tumor_name + ".svs")
    svs_image = openslide.open_slide(svs_path)

    print(CLAM_image.size)
    print(heatmap_image.size)

    width, height = CLAM_image.size
    
    # CLAM기준 가로, 세로의 patch개수 구하기
    print(tumor_name)
    width, height = int(width)/32, int(height)/32
    print(width, height)
    width = int(width)
    height = int(height)
    print(width, height)

    with torch.no_grad():
        for x in range(width-3):
            for y in range(height-3):
                x = 32 * x
                y = 32 * y
                total = 0
                wrong = 0
                for patch in os.listdir(patch_path):
                    if patch.endswith(".png"):
                        pattern = r"_(\d+)_(\d+)\.png"
                        matches = re.findall(pattern, patch)
                        patch_w, patch_h = matches[0]
                        patch_w, patch_h = int(patch_w)/8, int(patch_h)/8

                        if patch_w > x and patch_w < x + window_size:
                            total += 1
                            patch = Image.open(os.path.join(patch_path, patch))
                            image = transform(patch).to(device).unsqueeze(dim=0)

                            output = model(image)
                            probs_neg = sigmoid(output)[0][0].item()
                            probs_pos = sigmoid(output)[0][0].item()

                            if pos_or_neg == "pos":
                                if probs_neg >= 0.6:
                                    wrong += 1
                            else:
                                if probs_pos >= 0.6:
                                    wrong += 1
                
                if total != 0:
                    print(x, y, total, wrong)
                    acc = (total - wrong) / total * 100.
                    if acc <= 90:
                        crop(CLAM_image, x, y, "CLAM", output_path)
                        crop(heatmap_image, x, y, "heatmap", output_path)
                        crop_svs(svs_image, x, y, output_path)
        print("Done\n")
                            
                        


def crop(image, x, y, name, output_path):
    cropped_image = image.crop((x, y, x + window_size, y + window_size))
    output_path = os.path.join(output_path + f"_{x}_{y}_{name}.jpeg")
    cropped_image.save(output_path)

def crop_svs(image, x, y, output_path):
    level = 0
    x = x * 8
    y = y * 8
    window_size * 8
    region = image.read_region((x, y), level, (window_size, window_size))
    region = region.convert("RGB")
    output_path = os.path.join(output_path + f"_{x}_{y}_svs.jpeg")
    region.save(output_path, "JPEG")


with open("../tumorAnnotation.json", "r") as file:
    data = json.load(file)

keys_pos = ["POSITIVE_TEST_SSSF"]
keys_neg = ["NEGATIVE_TEST_SSSF"]

for key in keys_pos:
    current_list = data.get(key, [])
    for value in current_list:
        value = value.split('/')[-1]
        crop_region(value, "pos", output_path)


for key in keys_neg:
    current_list = data.get(key, [])
    for value in current_list:
        value = value.split('/')[-1]
        crop_region(value, "neg", output_path)
