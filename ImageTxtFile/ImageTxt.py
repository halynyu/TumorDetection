import json
import os
import sys

os.chdir('..')

with open('tumorAnnotation.json', 'r') as f:
    json_data = json.load(f)


pos_train_patch_path = []
neg_train_patch_path = []

pos_test_patch_path = []
neg_test_patch_path = []

pos_val_patch_path = []
neg_val_patch_path = []


for key, value in json_data.items():

    if key == 'POSITIVE_TRAIN_LUAC':
        pos_train_patch_path = value
    elif key == 'NEGATIVE_TRAIN_LUAC':
        neg_train_patch_path = value

    elif key == 'POSITIVE_TEST_LUAC':
        pos_test_patch_path = value
    elif key == 'NEGATIVE_TEST_LUAC':
        neg_test_patch_path = value

    elif key == 'POSITIVE_VALIDATION_LUAC':
        pos_val_patch_path = value
    elif key == 'NEGATIVE_VALIDATION_LUAC':
        neg_val_patch_path = value


with open(f'DatasetTxtFile/positive_train_LUAC_231114.txt', 'w') as f:
    for path in pos_train_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')

with open(f'DatasetTxtFile/negative_train_LUAC_231114.txt', 'w') as f:
    for path in neg_train_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')

with open(f'DatasetTxtFile/positive_test_LUAC_231114.txt', 'w') as f:
    for path in pos_test_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')

with open(f'DatasetTxtFile/negative_test_LUAC_231114.txt', 'w') as f:
    for path in neg_test_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')

with open(f'DatasetTxtFile/positive_validation_LUAC_231114.txt', 'w') as f:
    for path in pos_val_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')

with open(f'DatasetTxtFile/negative_validation_LUAC_231114.txt', 'w') as f:
    for path in neg_val_patch_path:
        for image_path in os.listdir(path):
            full_image_path = os.path.join(path, image_path)
            f.write(full_image_path + '\n')