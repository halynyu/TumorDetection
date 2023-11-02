from PIL import Image
import numpy as np
import openslide
import os
import re
import time
from tqdm import tqdm

JPEG_FOLDER = "JPEG_FOLDER"
ALK_NEGATIVE_TUMOR = [
    "./DATASET/LUAC/LUAC_1-1_A2.svs"
]
RED_THRESHOLD = 230
EXAMPLE_PATCH_PATH = "./sample_patch/10_LUAC_1-1_A2_x_2128_y_3816_a_99.524.png"
EXAMPLE_PATCH = Image.open("./sample_patch/10_LUAC_1-1_A2_x_2128_y_3816_a_99.524.png").convert('RGB')
PATCH_SIZE = EXAMPLE_PATCH.width
print(PATCH_SIZE)

def jpeg_svs_matching(svs_name, length, JPEG_FOLDER):
    for jpeg_name in os.listdir(JPEG_FOLDER):
        print(jpeg_name)
        if svs_name == jpeg_name[:length]:
            print(jpeg_name)
            return Image.open(os.path.join(JPEG_FOLDER, jpeg_name)).convert('RGB')
    
def jpeg_svs_ratio(svs_image, jpeg_image):
    ratio = svs_image.level_dimensions[0][0] / jpeg_image.width
    return ratio

def patching(slide, y, x, PATCH_SIZE, svs_jpeg_ratio):
    level = 0
    image = slide.read_region((y * svs_jpeg_ratio, x * svs_jpeg_ratio), level, (PATCH_SIZE, PATCH_SIZE))
    patch = np.array(image.convert('RGB'))
    return patch

def check_red_threshold(patch, patch_size, red_threshold):
    width, height = patch.size
    pixels = patch.load()
    total_red = 0
    for i in range(width):
        for j in range(height):
            r, _, _ = pixels[i, j]
            total_red += r
    average_red = total_red / (width * height)
    print(average_red)
    if average_red > RED_THRESHOLD:
        return True
        
    else:
        return False

    

def find_coordination(JPEG_IMAGE, example_patch_name):
    match = re.search(r"_x_(\d+)_y_(\d+)_", example_patch_name)
    if match:
        x_coordinate = int(match.group(1))
        y_coordinate = int(match.group(2))

    while x_coordinate > 0:
        x_coordinate -= PATCH_SIZE
    while y_coordinate > 0:
        y_coordinate -= PATCH_SIZE
    
    x_coordinate += PATCH_SIZE
    y_coordinate += PATCH_SIZE

    print(f"시작점 : ({x_coordinate}, {y_coordinate})")
    
    return x_coordinate, y_coordinate


def make_patch(NEGATIVE_FOLDER, JPEG_FOLDER):
    _length = len(NEGATIVE_FOLDER[0].split("/")[3])-4
    print(_length)
    SVS_IMAGE = openslide.open_slide(ALK_NEGATIVE_TUMOR[0])
    SVS_NAME = NEGATIVE_FOLDER[0].split("/")[3][:_length]
    print(SVS_IMAGE)
    print(SVS_NAME)

    JPEG_IMAGE = Image.open("./JPEG_FOLDER/LUAC_1-1_A2_blockmap.png")
    print(JPEG_IMAGE)
    EXAMPLE_PATCH_NAME = EXAMPLE_PATCH_PATH.split("/")[2]
    PATCH_SIZE = EXAMPLE_PATCH.width

    # SVS와 JPEG 이미지의 비율
    tau = int(jpeg_svs_ratio(SVS_IMAGE, JPEG_IMAGE))
    x_start, y_start = find_coordination(JPEG_IMAGE, EXAMPLE_PATCH_NAME)

    if not os.path.exists(f"PATCH_20231102/NEGATIVE_250/{SVS_NAME}"):
        os.makedirs(f"PATCH_20231102/NEGATIVE_250/{SVS_NAME}")

    start_time = time.time()
    fail_patch = 0
    success_patch = 0

    for h in tqdm(range(x_start, JPEG_IMAGE.height, PATCH_SIZE)):
        for w in range(y_start, JPEG_IMAGE.width, PATCH_SIZE):
            patch = JPEG_IMAGE.crop((h, w, h+PATCH_SIZE, w+PATCH_SIZE))
            if check_red_threshold(patch, PATCH_SIZE, RED_THRESHOLD):
                success_patch += 1
                print(h, w)
                print(tau)
                patch_ = patching(SVS_IMAGE, h, w, PATCH_SIZE, tau)
                filename = f"{SVS_NAME}_{h}_{w}.png"
                patch_save_folder_name = f"PATCH_20231102/NEGATIVE_250/{SVS_NAME}"
                Image.fromarray(patch_).save(os.path.join(patch_save_folder_name, filename))
            else:
                fail_patch += 1

    end_time = time.time()
    print(f"{SVS_NAME} Patch 생성 소요시간 : {end_time - start_time}")
    print(JPEG_IMAGE.width / PATCH_SIZE)
    print(f"성공 : {success_patch}, 실패 : {fail_patch}")

make_patch(ALK_NEGATIVE_TUMOR, JPEG_FOLDER="JPEG_FOLDER")