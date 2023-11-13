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
# Making Datasets in different Folder
from makeDataset import *

from tqdm import tqdm
import time

from classification_model import make_ResNet
from image_make_utils import make_HeatMap
from train_test import train, eval



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
    parser.add_argument('--model_save_path', type=str, default=f'Model_save/20231109_model6',
                        help="Where to Save model ")
    parser.add_argument('--pretrained', type=bool,  default=False,
                        help="Using Pretrained or not")
    parser.add_argument('--loss_image_save', type=bool, default =False,
                        help="Sampling Image getting High Loss ")
    args = parser.parse_args()

    return args


if __name__ == '__main__':


    args = get_args()

    # Define model, criterion, optimizer, batch_size, epoch


    gpu_count = torch.cuda.device_count()

    # dist.init_process_group(backend='nccl')


    model = make_ResNet(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("**************** Model Architecture **********************************\n")
    print(model)
    print("********************************************************************\n\n")

    model.to(device)


    # DataLoader

    # # Train용 전체 dataset 들어간 DataLoader
    # train_Dataloader = DataLoader(concat_Dataset, batch_size = args.batch_size, shuffle =shuffle,
    #                         pin_memory = pin_memory)

    # valid_Dataloader = DataLoader(concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
    #                         pin_memory = pin_memory)
    # test_Dataloader = DataLoader(concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
    #                         pin_memory = pin_memory)



    # Test용
    # LUAC
    LUAC_train_Dataloader = DataLoader(LUAC_concat_Dataset, batch_size = args.batch_size, shuffle=shuffle,
                                pin_memory = pin_memory)
    LUAC_valid_Dataloader = DataLoader(LUAC_concat_Valid_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)
    LUAC_test_Dataloader = DataLoader(LUAC_concat_Test_Dataset, batch_size = args.batch_size, shuffle = shuffle,
                            pin_memory = pin_memory)

    # # YS
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

    criterion_weight = [[1.8,1]]

    for weights in criterion_weight:

        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path, exist_ok=True)
        pos, neg = weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # train(model, train_Dataloader, valid_Dataloader, criterion, optimizer, args.epochs, args.model_save_path, device, weights)

        # print("Complete !!")

    if args.pretrained :
        checkpoint = torch.load(f'/home/lab/Tumor_Detection/Model_save/20231109_model6/1.8_1/epoch_19_all.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    print("LUAC accuracy")
    eval(model, LUAC_test_Dataloader, criterion, device=device, batch_size=args.batch_size, image_save=False, dataset=LUAC_concat_Test_Dataset)
    print("TCGA accuracy")
    eval(model, TCGA_test_Dataloader, criterion, device=device, batch_size=args.batch_size, image_save=False, dataset=TCGA_concat_Test_Dataset)
    print("YS accuracy")
    eval(model, YS_test_Dataloader, criterion, device=device, batch_size=args.batch_size, image_save=False, dataset=YS_concat_Test_Dataset)
    print("SSSF accuracy")
    eval(model, SSSF_test_Dataloader, criterion, device=device, batch_size=args.batch_size, image_save=False, dataset=SSSF_concat_Test_Dataset)


    # make_HeatMap(model, device)