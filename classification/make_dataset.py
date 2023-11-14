from dataset import *
from PIL import Image
import os
from torch.utils.data import ConcatDataset
import torch.distributed as dist
import random

os.chdir('..')

# DATA PATH

# LUAC
positive_train_LUAC_file_path = f"DatasetTxtFile/positive_train_LUAC_231114.txt"
negative_train_LUAC_file_path = f"DatasetTxtFile/negative_train_LUAC_231114.txt"

pos_train_LUAC_data_path = []
neg_train_LUAC_data_path = []

positive_test_LUAC_file_path = f"DatasetTxtFile/positive_test_LUAC_231114.txt"
negative_test_LUAC_file_path = f"DatasetTxtFile/negative_test_LUAC_231114.txt"

pos_test_LUAC_data_path = []
neg_test_LUAC_data_path = []

positive_validation_LUAC_file_path = f"DatasetTxtFile/positive_validation_LUAC_231114.txt"
negative_validation_LUAC_file_path = f"DatasetTxtFile/negative_validation_LUAC_231114.txt"

pos_valid_LUAC_data_path = []
neg_valid_LUAC_data_path = []

# TCGA
positive_train_TCGA_file_path = f"DatasetTxtFile/positive_train_TCGA_231114.txt"
negative_train_TCGA_file_path = f"DatasetTxtFile/negative_train_TCGA_231114.txt"

pos_train_TCGA_data_path = []
neg_train_TCGA_data_path = []

positive_test_TCGA_file_path = f"DatasetTxtFile/positive_test_TCGA_231114.txt"
negative_test_TCGA_file_path = f"DatasetTxtFile/negative_test_TCGA_231114.txt"

pos_test_TCGA_data_path = []
neg_test_TCGA_data_path = []

positive_validation_TCGA_file_path = f"DatasetTxtFile/positive_validation_TCGA_231114.txt"
negative_validation_TCGA_file_path = f"DatasetTxtFile/negative_validation_TCGA_231114.txt"

pos_valid_TCGA_data_path = []
neg_valid_TCGA_data_path = []

# YS
positive_train_YS_file_path = f"DatasetTxtFile/positive_train_YS_231114.txt"
negative_train_YS_file_path = f"DatasetTxtFile/negative_train_YS_231114.txt"

pos_train_YS_data_path = []
neg_train_YS_data_path = []

positive_test_YS_file_path = f"DatasetTxtFile/positive_test_YS_231114.txt"
negative_test_YS_file_path = f"DatasetTxtFile/negative_test_YS_231114.txt"

pos_test_YS_data_path = []
neg_test_YS_data_path = []

positive_validation_YS_file_path = f"DatasetTxtFile/positive_validation_YS_231114.txt"
negative_validation_YS_file_path = f"DatasetTxtFile/negative_validation_YS_231114.txt"

pos_valid_YS_data_path = []
neg_valid_YS_data_path = []


# SSSF
positive_train_SSSF_file_path = f"DatasetTxtFile/positive_train_SSSF_231114.txt"
negative_train_SSSF_file_path = f"DatasetTxtFile/negative_train_SSSF_231114.txt"

pos_train_SSSF_data_path = []
neg_train_SSSF_data_path = []

positive_test_SSSF_file_path = f"DatasetTxtFile/positive_test_SSSF_231114.txt"
negative_test_SSSF_file_path = f"DatasetTxtFile/negative_test_SSSF_231114.txt"

pos_test_SSSF_data_path = []
neg_test_SSSF_data_path = []

positive_validation_SSSF_file_path = f"DatasetTxtFile/positive_validation_SSSF_231114.txt"
negative_validation_SSSF_file_path = f"DatasetTxtFile/negative_validation_SSSF_231114.txt"

pos_valid_SSSF_data_path = []
neg_valid_SSSF_data_path = []






# DATA OPEN

# LUAC
with open(positive_train_LUAC_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        pos_train_LUAC_data_path.append(line)

with open(negative_train_LUAC_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_train_LUAC_data_path.append(line)

with open(positive_test_LUAC_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_test_LUAC_data_path.append(line)

with open(negative_test_LUAC_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_test_LUAC_data_path.append(line)

with open(positive_validation_LUAC_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_valid_LUAC_data_path.append(line)

with open(negative_validation_LUAC_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_valid_LUAC_data_path.append(line)


# TCGA
with open(positive_train_TCGA_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        pos_train_TCGA_data_path.append(line)

with open(negative_train_TCGA_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_train_TCGA_data_path.append(line)

with open(positive_test_TCGA_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_test_TCGA_data_path.append(line)

with open(negative_test_TCGA_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_test_TCGA_data_path.append(line)

with open(positive_validation_TCGA_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_valid_TCGA_data_path.append(line)

with open(negative_validation_TCGA_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_valid_TCGA_data_path.append(line)

# YS
with open(positive_train_YS_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        pos_train_YS_data_path.append(line)

with open(negative_train_YS_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_train_YS_data_path.append(line)

with open(positive_test_YS_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_test_YS_data_path.append(line)

with open(negative_test_YS_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_test_YS_data_path.append(line)

with open(positive_validation_YS_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_valid_YS_data_path.append(line)

with open(negative_validation_YS_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_valid_YS_data_path.append(line)

# SSSF
with open(positive_train_SSSF_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        pos_train_SSSF_data_path.append(line)

with open(negative_train_SSSF_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_train_SSSF_data_path.append(line)

with open(positive_test_SSSF_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_test_SSSF_data_path.append(line)

with open(negative_test_SSSF_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_test_SSSF_data_path.append(line)

with open(positive_validation_SSSF_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        pos_valid_SSSF_data_path.append(line)

with open(negative_validation_SSSF_file_path, 'r') as f:
    for line in f :
        line = line.strip()
        neg_valid_SSSF_data_path.append(line)

transform = ImageTransform()

def select_data(negative_list, positive_list, tau):
    random.seed(42)
    random.shuffle(negative_list)
    list_len = len(positive_list) * tau
    return negative_list[:list_len]


# [If Necessary] Neg, Pos Dataset 불균형이 심하다면 tau값 조정하여 Neg : Pos dataset 개수 비율 맞추기
# tau = 1
# neg_train_data_path = select_data(neg_train_data_path, pos_train_data_path, tau)
# print(len(neg_train_data_path))

# def make_dataset(pt, nt):
#     random.seed(42)
#     random.shuffle(pt)
#     random.shuffle(nt)

#     pos_length, neg_length = int(len(pt) * 0.1), int(len(nt) * 0.1)
#     return pt[pos_length:], nt[neg_length:], pt[:pos_length], nt[:neg_length]

# pos_train_data_path, neg_train_data_path, pos_valid_data_path, neg_valid_data_path = make_dataset(pos_train_data_path, neg_train_data_path)

def data_info(pos_train_data_path, neg_train_data_path, pos_test_data_path, neg_test_data_path, pos_valid_data_path, neg_valid_data_path):
    
    train, test, valid = len(pos_train_data_path) + len(neg_train_data_path), len(pos_test_data_path) + len(neg_test_data_path), len(pos_valid_data_path) + len(neg_valid_data_path)

    print(f"TRAIN  | {train}     VALID | {valid}       TEST | {test}")
    print(f"ALK+    |   TRAIN : {len(pos_train_data_path)}      VALID : {len(pos_valid_data_path)}      TEST : {len(pos_test_data_path)}")
    print(f"ALK-    |   TRAIN : {len(neg_train_data_path)}      VALID : {len(neg_valid_data_path)}      TEST : {len(neg_test_data_path)}")

print("\n")
print("LUAC Data Distribution")
data_info(pos_train_LUAC_data_path, neg_train_LUAC_data_path, pos_test_LUAC_data_path, neg_test_LUAC_data_path, pos_valid_LUAC_data_path, neg_valid_LUAC_data_path)
print("\n")
print("TCGA Data Distribution")
data_info(pos_train_TCGA_data_path, neg_train_TCGA_data_path, pos_test_TCGA_data_path, neg_test_TCGA_data_path, pos_valid_TCGA_data_path, neg_valid_TCGA_data_path)
print("\n")
print("YS Data Distribution")
data_info(pos_train_YS_data_path, neg_train_YS_data_path, pos_test_YS_data_path, neg_test_YS_data_path, pos_valid_YS_data_path, neg_valid_YS_data_path)
print("\n")
print("SSSF Data Distribution")
data_info(pos_train_SSSF_data_path, neg_train_SSSF_data_path, pos_test_SSSF_data_path, neg_test_SSSF_data_path, pos_valid_SSSF_data_path, neg_valid_SSSF_data_path)


# Concat Dataset

# LUAC
LUACPosTrainDataset = TumorDataset(pos_train_LUAC_data_path, 1,transform)
LUACNegTrainDataset = TumorDataset(neg_train_LUAC_data_path, 0,transform)

LUAC_concat_Dataset = ConcatDataset([LUACPosTrainDataset, LUACNegTrainDataset])

LUACPosTestDataset = TumorDataset(pos_test_LUAC_data_path, 1,transform)
LUACNegTestDataset = TumorDataset(neg_test_LUAC_data_path, 0,transform)

LUAC_concat_Test_Dataset = ConcatDataset([LUACPosTestDataset, LUACNegTestDataset])

LUACPosValidDataset = TumorDataset(pos_valid_LUAC_data_path, 1, transform)
LUACNegValidDataset = TumorDataset(neg_valid_LUAC_data_path, 0, transform)

LUAC_concat_Valid_Dataset = ConcatDataset([LUACPosValidDataset, LUACNegValidDataset])

# TCGA
TCGAPosTrainDataset = TumorDataset(pos_train_TCGA_data_path, 1,transform)
TCGANegTrainDataset = TumorDataset(neg_train_TCGA_data_path, 0,transform)

TCGA_concat_Dataset = ConcatDataset([TCGAPosTrainDataset, TCGANegTrainDataset])

TCGAPosTestDataset = TumorDataset(pos_test_TCGA_data_path, 1,transform)
TCGANegTestDataset = TumorDataset(neg_test_TCGA_data_path, 0,transform)

TCGA_concat_Test_Dataset = ConcatDataset([TCGAPosTestDataset, TCGANegTestDataset])

TCGAPosValidDataset = TumorDataset(pos_valid_TCGA_data_path, 1, transform)
TCGANegValidDataset = TumorDataset(neg_valid_TCGA_data_path, 0, transform)

TCGA_concat_Valid_Dataset = ConcatDataset([TCGAPosValidDataset, TCGANegValidDataset])

# YS
YSPosTrainDataset = TumorDataset(pos_train_YS_data_path, 1,transform)
YSNegTrainDataset = TumorDataset(neg_train_YS_data_path, 0,transform)

YS_concat_Dataset = ConcatDataset([YSPosTrainDataset, YSNegTrainDataset])

YSPosTestDataset = TumorDataset(pos_test_YS_data_path, 1,transform)
YSNegTestDataset = TumorDataset(neg_test_YS_data_path, 0,transform)

YS_concat_Test_Dataset = ConcatDataset([YSPosTestDataset, YSNegTestDataset])

YSPosValidDataset = TumorDataset(pos_valid_YS_data_path, 1, transform)
YSNegValidDataset = TumorDataset(neg_valid_YS_data_path, 0, transform)

YS_concat_Valid_Dataset = ConcatDataset([YSPosValidDataset, YSNegValidDataset])

# SSSF
SSSFPosTrainDataset = TumorDataset(pos_train_SSSF_data_path, 1,transform)
SSSFNegTrainDataset = TumorDataset(neg_train_SSSF_data_path, 0,transform)

SSSF_concat_Dataset = ConcatDataset([SSSFPosTrainDataset, SSSFNegTrainDataset])

SSSFPosTestDataset = TumorDataset(pos_test_SSSF_data_path, 1,transform)
SSSFNegTestDataset = TumorDataset(neg_test_SSSF_data_path, 0,transform)

SSSF_concat_Test_Dataset = ConcatDataset([SSSFPosTestDataset, SSSFNegTestDataset])

SSSFPosValidDataset = TumorDataset(pos_valid_SSSF_data_path, 1, transform)
SSSFNegValidDataset = TumorDataset(neg_valid_SSSF_data_path, 0, transform)

SSSF_concat_Valid_Dataset = ConcatDataset([SSSFPosValidDataset, SSSFNegValidDataset])


concat_Dataset = ConcatDataset([LUAC_concat_Dataset, SSSF_concat_Dataset, TCGA_concat_Dataset, YS_concat_Dataset])
concat_Valid_Dataset = ConcatDataset([LUAC_concat_Valid_Dataset, SSSF_concat_Valid_Dataset, TCGA_concat_Valid_Dataset, YS_concat_Valid_Dataset])
concat_Test_Dataset = ConcatDataset([LUAC_concat_Test_Dataset, SSSF_concat_Test_Dataset, TCGA_concat_Test_Dataset, YS_concat_Test_Dataset])



shuffle = True
pin_memory = True # Memory에 Data를 바로 올릴수있도록 하는 옵션

print("\n")
print("Dataset creation completed !")