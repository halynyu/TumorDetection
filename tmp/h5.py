import h5py
import openslide
from PIL import Image

file_path = "./LUAC_1-1_A4.h5"
blockmap_file_path = "./LUAC_1-1_A4_blockmap.h5"

svs_file_path = ["./LUAC_1-1_A4.svs"]
png_img = Image.open("./LUAC_1-1_A4_blockmap.png")


slide = openslide.open_slide(svs_file_path[0])
print(slide.dimensions)
print(png_img.width, png_img.height)

print("svs 기준")
with h5py.File(file_path, 'r') as file:
    # 'coords'에 해당하는 데이터셋 읽기
    coords_dataset = file['coords']
    
    # 데이터셋 내용 출력
    coords_data = coords_dataset[:]
    print(f"Contents of 'coords' dataset: {coords_data}")
    num_coords = coords_dataset.shape[0]
    
    print(f"The number of 'coords' datasets: {num_coords}")
