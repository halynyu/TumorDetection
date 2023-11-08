from PIL import Image
import h5py
import openslide
import os

PNG_PATH = "LUAC_1-1_A4_blockmap.png"

coords_list = [(1520, 5120), (1776, 4608), (696, 6880), (13496, 6880), (13496, 7136), (13496, 7392)]

def patching(input_file, output_file, coords_list, patch_size, tau):
    image = Image.open(input_file).convert('RGB')

    for i, (start_x, start_y) in enumerate(coords_list):
        start_x = int(start_x/8)
        start_y = int(start_y/8)
        print(start_x, start_y)
        patch = image.crop((start_x, start_y, start_x + patch_size, start_y + patch_size)).convert('RGB')
        output_path = f"{output_file}/patch_{start_x*8}_{start_y*8}.png"
        patch.save(output_path)

input_image_path = "LUAC_1-1_A4_blockmap.png"
output_patches_folder = "."
patch_size = 32
tau = 8

os.makedirs(output_patches_folder, exist_ok = True)

patching(input_image_path, output_patches_folder, coords_list, patch_size, tau)