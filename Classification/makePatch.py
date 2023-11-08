from PIL import Image, ImageDraw, ImageCms
import h5py
import openslide
import os
from tqdm import tqdm
import time
import numpy as np

svs_images = [
# "SS18-19871_14000_10000", "SS18-19871_14000_30000", "SS18-19871_14000_50000", "SS18-19871_34000_40000", "SS18-19871_34000_60000", "SS18-19871_54000_10000", "SS18-19871_54000_30000", 
# "SS18-19871_54000_50000", "SS18-19871_74000_20000", "SS18-19871_74000_40000", 

# "SS21-26050_5000_0", "SS21-26050_5000_20000", "SS21-26050_5000_40000", "SS21-26050_5000_60000", "SS21-26050_25000_0", "SS21-26050_25000_20000", "SS21-26050_25000_40000", 
# "SS21-26050_25000_60000", "SS21-26050_45000_0", "SS21-26050_45000_20000", "SS21-26050_45000_40000", "SS21-26050_45000_60000", "SS21-26050_65000_0", "SS21-26050_65000_20000", 
# "SS21-26050_65000_40000", "SS21-26050_65000_60000", "SS21-26050_85000_0", "SS21-26050_85000_20000", "SS21-26050_85000_40000", "SS21-26050_85000_60000",

# "SS22-6559_9000_9000", "SS22-6559_9000_29000", "SS22-6559_29000_9000", "SS22-6559_29000_29000", "SS22-6559_49000_9000", "SS22-6559_49000_29000", 

# "SS22-26507_55000_23000", "SS22-26507_55000_43000", "SS22-26507_55000_63000", "SS22-26507_75000_3000", "SS22-26507_75000_23000", "SS22-26507_75000_43000", "SS22-26507_75000_63000", "SS22-26507_95000_3000", "SS22-26507_95000_23000", 


# "YS12-1606_0_23000", "YS12-1606_0_43000", "YS12-1606_0_63000", "YS12-1606_20000_23000", "YS12-1606_20000_43000", "YS12-1606_20000_63000", 

# "YS15-1117_0_25000", "YS15-1117_0_45000", "YS15-1117_0_65000", "YS15-1117_20000_25000", "YS15-1117_20000_45000", "YS15-1117_20000_65000", 

# "YS16-2177_0_20000", "YS16-2177_0_40000", "YS16-2177_0_60000", "YS16-2177_20000_20000", "YS16-2177_20000_40000", "YS16-2177_20000_60000", 

# "YS17-2284_0_3000", "YS17-2284_0_23000", "YS17-2284_0_43000", "YS17-2284_20000_3000", "YS17-2284_20000_23000", "YS17-2284_20000_43000", 

# "YS18-1737_0_25000", "YS18-1737_0_45000", "YS18-1737_0_65000", "YS18-1737_20000_25000", "YS18-1737_20000_45000", "YS18-1737_20000_65000", 

# "YS18-3319_0_0", "YS18-3319_0_20000", "YS18-3319_0_40000", "YS18-3319_20000_0", "YS18-3319_20000_20000", "YS18-3319_20000_40000", 

# "YS19-0568_0_0", "YS19-0568_0_20000", "YS19-0568_0_40000", "YS19-0568_20000_0", "YS19-0568_20000_20000", "YS19-0568_20000_40000", 

# "YS19-0946_0_35000", "YS19-0946_0_55000", "YS19-0946_20000_35000", "YS19-0946_20000_55000", 

# "YS19-2439_0_20000", "YS19-2439_0_40000", "YS19-2439_0_60000", "YS19-2439_20000_20000", "YS19-2439_20000_40000", "YS19-2439_20000_60000", 

# "YS20-0561_0_15000", "YS20-0561_0_35000", "YS20-0561_0_55000", "YS20-0561_20000_15000", "YS20-0561_20000_35000", "YS20-0561_20000_55000", 

# "YS20-2317_0_0", "YS20-2317_0_20000", "YS20-2317_0_40000", "YS20-2317_20000_0", "YS20-2317_20000_20000", "YS20-2317_20000_40000"


# "TCGA-67-6216-01Z-00-DX1--a", "TCGA-67-6215-01A-01-TS1--a", "TCGA-78-7163-01A-01-TS1--a", 
"TCGA-86-A4P8-01Z-00-DX1--a"
]

# base_path : CLAM의 결과 (png, h5 파일이 있는 directory)
base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/heatmap_raw_results/HEATMAP_OUTPUT/ POS/"
# svs_base_path : svs 파일이 있는 directory
svs_base_path = "/home/lab/Tumor_Detection/CLAM/heatmaps/demo/slides/"



def extract_coords(file_path):
    with h5py.File(file_path, 'r') as file:
        coords_dataset = file['coords']
        coords_data = coords_dataset[:]
        return coords_data

def check_red_threshold(patch, patch_size, red_threshold):
    LAB_image = patch.convert("LAB")
    _, a, _ = LAB_image.split()

    average_a = sum(a.getdata()) / (patch.width * patch.height)

    if average_a > red_threshold:
        return True
    else:
        return False






def patching(slide, y, x, PATCH_SIZE, tau):
    level = 0
    image = slide.read_region((y * tau, x * tau,), level, (PATCH_SIZE, PATCH_SIZE))
    patch = np.array(image.convert('RGB'))
    return patch


def make_patch(file_list, patch_size, RED_THRESHOLD):
    for svs_name in file_list:

        if not os.path.exists(f"PATCH_h5/{svs_name}"):
            os.makedirs(f"PATCH_h5/{svs_name}")
        
        output_path = f"PATCH_h5/{svs_name}"
        svs_file = openslide.open_slide(os.path.join(svs_base_path + svs_name + ".svs"))
        blockmap_file = Image.open(os.path.join(base_path, svs_name, svs_name + "_blockmap.png"))
    
        h5_file = os.path.join(base_path, svs_name, svs_name + ".h5")
        coords_list = extract_coords(h5_file)

        tau = int(svs_file.level_dimensions[0][0] / blockmap_file.width)

        print(f"\n{svs_name}의 patch 개수 : {coords_list.shape[0]}")

        success_patch = 0
        fail_patch = 0


        draw = ImageDraw.Draw(blockmap_file)
        start_time = time.time()

        for i, (start_x, start_y) in enumerate(coords_list):
            start_x = int(start_x / tau)
            start_y = int(start_y / tau)



            patch = blockmap_file.crop((start_x, start_y, start_x + patch_size, start_y + patch_size)).convert('RGB')
            
            if check_red_threshold(patch, patch_size, RED_THRESHOLD):
                success_patch += 1
                patch_ = patching(svs_file, start_y, start_x, patch_size, tau)
                filename = f"{svs_name}_{start_x}_{start_y}.png"
                Image.fromarray(patch_).save(os.path.join(output_path, filename))
                draw.rectangle((start_x, start_y, start_x + patch_size, start_y + patch_size), outline=(0, 0, 0))

            else:
                fail_patch += 1
    
        end_time = time.time()
        blockmap_file.save(f"./heatmap/{svs_name}_heatmap.jpg")
        print(f"{svs_name} patch 생성 소요시간 : {end_time - start_time}")
        if success_patch + fail_patch == coords_list.shape[0]:
            print(f"Patch 개수 : {success_patch} / {coords_list.shape[0]}")
        else:
            print("error")


patch_size = 32
RED_THRESHOLD = 170


make_patch(svs_images, patch_size, RED_THRESHOLD)


