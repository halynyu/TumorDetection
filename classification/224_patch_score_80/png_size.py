import os
from PIL import Image

def get_png_sizes(directory_path = "./YS12-1606_0_23000"):
    png_sizes = {}

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".png"):
            file_path = os.path.join(directory_path, filename)
        img = Image.open(file_path)
        width, height = img.size
        png_sizes[filename] = (width, height)

current_directory = os.getcwd()
png_sizes = get_png_sizes(current_directory)

for filename, size in png_sizes.items():
    print(f"{size[1]}")
