from PIL import Image
import os

def get_png_size(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None

file_path = "CLAM/heatmaps/heatmap_production_results/HEATMAP_OUTPUT/sampled_patches/label_ POS_pred_0/topk_high_attention/0_LUAC_1-1_A2_x_1104_y_5096_a_100.000.png"
size = get_png_size(file_path)

if size:
    print(f"{size[0]}X{size[1]}")