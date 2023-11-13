import os

folder_path = "./"

with open('folder_list.txt', 'w') as file:
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            file.write(folder_name + '\n')