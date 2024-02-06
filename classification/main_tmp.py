import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision
import argparse
import os
import wandb
# Making Datasets in different Folder
from make_dataset import *

from tqdm import tqdm
import time

from classification_model import *
from image_make_utils import make_HeatMap
from train_test import train, criterion_optimizer, eval



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
    parser.add_argument('--model_save_path', type=str, default=f'Model_save/20230205_model1',
                        help="Where to Save model ")
    parser.add_argument('--pretrained', type=str,  default=False,
                        help="Using Pretrained or not")
    parser.add_argument('--loss_image_save', type=bool, default =False,
                        help="Sampling Image getting High Loss ")
    args = parser.parse_args()

    return args

# 이미지 로드 및 MobileNet에서 사용하는 224*224로 크기 변경 (전처리)
def preprocessing_image(img_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
    img = Image.open(img_path)
    img_transform(img)
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':


    args = get_args()

    # Define model, criterion, optimizer, batch_size, epoch


    gpu_count = torch.cuda.device_count()

    # dist.init_process_group(backend='nccl')

    # TODO
    # 

    #model = make_ResNet(args)
    model = make_MobileNetV2(args)

    device = torch.device('cuda:0')
    # print("**************** Model Architecture **********************************\n")
    # print(model)
    # print("********************************************************************\n\n")

    # print(device)
    model.to(device=device)


    # DataLoader

    # # Train용 전체 dataset 들어간 DataLoader
    train_Dataloader = DataLoader(concat_Dataset, batch_size = args.batch_size, shuffle =shuffle,
                           pin_memory = pin_memory)

    valid_Dataloader = DataLoader(concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                           pin_memory = pin_memory)
    test_Dataloader = DataLoader(concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                           pin_memory = pin_memory)



    # Test용
    # # LUAC
    LUAC_train_Dataloader = DataLoader(LUAC_concat_Dataset, batch_size = args.batch_size, shuffle=shuffle,
                                pin_memory = pin_memory)
    LUAC_valid_Dataloader = DataLoader(LUAC_concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    LUAC_test_Dataloader = DataLoader(LUAC_concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    LUAC_weights = [10,1]

    # YS
    YS_train_Dataloader = DataLoader(YS_concat_Dataset, batch_size = args.batch_size, shuffle=shuffle,
                                pin_memory = pin_memory)
    YS_valid_Dataloader = DataLoader(YS_concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    YS_test_Dataloader = DataLoader(YS_concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)

    # TCGA
    TCGA_train_Dataloader = DataLoader(TCGA_concat_Dataset, batch_size = args.batch_size, shuffle=shuffle,
                                pin_memory = pin_memory)
    TCGA_valid_Dataloader = DataLoader(TCGA_concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    TCGA_test_Dataloader = DataLoader(TCGA_concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)

    # SSSF
    SSSF_train_Dataloader = DataLoader(SSSF_concat_Dataset, batch_size = args.batch_size, shuffle=shuffle,
                                pin_memory = pin_memory)
    SSSF_valid_Dataloader = DataLoader(SSSF_concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    SSSF_test_Dataloader = DataLoader(SSSF_concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    
    weights = [1, 1]

    print(args.pretrained)


    # pretrained 인자를 False로 받았을 때 / Train, validation
    if args.pretrained == "False":
        print("Train")
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path, exist_ok=True)
        
        LUAC_criterion, LUAC_optimizer = criterion_optimizer(model, LUAC_weights)
        criterion, optimizer = criterion_optimizer(model, weights)
        train(model, LUAC_train_Dataloader, train_Dataloader, valid_Dataloader, LUAC_valid_Dataloader, YS_valid_Dataloader, TCGA_valid_Dataloader, SSSF_valid_Dataloader, LUAC_criterion, LUAC_optimizer, criterion, optimizer, args.epochs, args.model_save_path, device, LUAC_weights, weights)

        print("Complete !!")


    # pretrained 인자를 True로 받았을 때 / Test
    if args.pretrained == "True":
        print("Test")

        LUAC_criterion, LUAC_optimizer = criterion_optimizer(model, LUAC_weights)
        criterion, optimizer = criterion_optimizer(model, weights)

        # Test할 model의 위치를 수정하며 실험
        test_model_path = "/home/lab/Tumor_Detection/Model_save/20231228_model1/1_1/1_1/epoch_511_all.tar"

        print("LUAC")
        LUAC_total_pos, LUAC_correct_pos, LUAC_total_neg, LUAC_correct_neg, LUAC_pos_acc, LUAC_neg_acc, LUAC_total_acc = eval(model, LUAC_test_Dataloader, LUAC_criterion, LUAC_optimizer, device, args.batch_size, test_model_path)
        print(f"Positive | correct : {LUAC_correct_pos} , total : {LUAC_total_pos} | positive accuracy : {LUAC_pos_acc}")
        print(f"Negative | correct : {LUAC_correct_neg} , total : {LUAC_total_neg} | negative accuracy : {LUAC_neg_acc}")
        print(f"Total accuracy : {LUAC_total_acc}")

        print("YS")
        YS_total_pos, YS_correct_pos, YS_total_neg, YS_correct_neg, YS_pos_acc, YS_neg_acc, YS_total_acc = eval(model, YS_test_Dataloader, criterion, optimizer, device, args.batch_size, test_model_path)
        print(f"Positive | correct : {YS_correct_pos} , total : {YS_total_pos} | positive accuracy : {YS_pos_acc}")
        print(f"Negative | correct : {YS_correct_neg} , total : {YS_total_neg} | negative accuracy : {YS_neg_acc}")
        print(f"Total accuracy : {YS_total_acc}")

        print("TCGA")
        TCGA_total_pos, TCGA_correct_pos, TCGA_total_neg, TCGA_correct_neg, TCGA_pos_acc, TCGA_neg_acc, TCGA_total_acc = eval(model, TCGA_test_Dataloader, criterion, optimizer, device, args.batch_size, test_model_path)
        print(f"Positive | correct : {TCGA_correct_pos} , total : {TCGA_total_pos} | positive accuracy : {TCGA_pos_acc}")
        print(f"Negative | correct : {TCGA_correct_neg} , total : {TCGA_total_neg} | negative accuracy : {TCGA_neg_acc}")
        print(f"Total accuracy : {TCGA_total_acc}")

        print("SSSF")
        SSSF_total_pos, SSSF_correct_pos, SSSF_total_neg, SSSF_correct_neg, SSSF_pos_acc, SSSF_neg_acc, SSSF_total_acc = eval(model, SSSF_test_Dataloader, criterion, optimizer, device, args.batch_size, test_model_path)
        print(f"Positive | correct : {SSSF_correct_pos} , total : {SSSF_total_pos} | positive accuracy : {SSSF_pos_acc}")
        print(f"Negative | correct : {SSSF_correct_neg} , total : {SSSF_total_neg} | negative accuracy : {SSSF_neg_acc}")
        print(f"Total accuracy : {SSSF_total_acc}")



        total_pos = LUAC_total_pos + YS_total_pos + SSSF_total_pos + TCGA_total_pos
        correct_pos = LUAC_correct_pos + YS_correct_pos + SSSF_correct_pos + TCGA_correct_pos
        total_neg = LUAC_total_neg + YS_total_neg + SSSF_total_neg + TCGA_total_neg
        correct_neg = LUAC_correct_neg + YS_correct_neg + SSSF_correct_neg + TCGA_correct_neg

        total_pos_acc = 100. * correct_pos / total_pos
        total_neg_acc = 100. * correct_neg / total_neg
        total_acc = 100. * (correct_pos + correct_neg) / (total_pos + total_neg)

        print("Total")
        print(f"Positive accuracy : {total_pos_acc} | Negative accuray : {total_neg_acc}\nTotal Test accuracy : {total_acc}") 


    # make_HeatMap(model, device)
