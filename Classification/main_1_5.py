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
from train_test_tmp import train, eval



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
    parser.add_argument('--model_save_path', type=str, default=f'Model_save/2023_10_1_6',
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

    # Train용 전체 dataset 들어간 DataLoader
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

    criterion_weight = [[5,1]]

    for weights in criterion_weight:

        # Model을 여기서 다시 초기화 해줘야할듯?
        # 1. 첫번째 가능성 ->예를들어, weight 1.5 :1 로했을 때 훈련시킨 모델에 대해서 바로 1.6:1 로 이어서 훈련하는 문제 때문인>거일수도..
        # 2. 각 훈련당 20 Iteration에서 Loss가 급격하게 변화가 생김 -> Weight가 중첩된듯
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path, exist_ok=True)
        pos, neg = weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # train(model, train_Dataloader, valid_Dataloader, criterion, optimizer, args.epochs, args.model_save_path, device, weights)

        # print("Complete !!")

    if args.pretrained :
        checkpoint = torch.load(f'/home/lab/Tumor_Detection/Model_save/2023_10_1_6/5_1/epoch_0_all.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    # eval(model, LUAC_test_Dataloader, criterion, device=device, image_save=False)
    # eval(model, TCGA_test_Dataloader, criterion, device=device, image_save=False)
    # eval(model, YS_test_Dataloader, criterion, device=device, image_save=False)

    make_HeatMap(model, device)