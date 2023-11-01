import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import time
import os

def train(model, dataloader, valid_dataloader, criterion, optimizer, epochs, save_path, device, weights):
    BEST_LOSS = 1e11
    model.to(device)
    model.train()
    print(device)
    criterion.to(device)

    save_path = os.path.join(save_path, f'{weights[0]}_{weights[1]}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        running_loss = 0.0
        num = 1
        for i, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss= criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num += 1
            if num % 50 == 2:
                print(f"{i}th Iteration Loss : {loss.item()} ")

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(torch.sigmoid(model(x.to(device))), torch.eye(2, device=device)[y.to(device)]).item() for x, y in valid_dataloader)
            if BEST_LOSS > valid_loss:
                BEST_LOSS = valid_loss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(save_path,f'epoch_{epoch}_all.tar'))


        end_time = time.time()
        print(f" EPOCH : {epoch + 1} / {epochs} | Validation Loss per Epoch : {valid_loss / len(valid_dataloader)} | Training Time per epoch : {end_time - start_time}s ",end='\n')


def eval(model, dataloader, criterion, device, batch_size, image_save=False):
    model.to(device)
    model.eval()
    criterion.to(device)


    min_loss = 1e9
    min_loss_image = None
    test_loss = 0
    correct = 0
    batch_count = 0
    total = 0

    save_path = 'evaluation_image/1/'
    if image_save :
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    num_ = 0

    with torch.no_grad():
        import torchvision
        import numpy as np
        from PIL import Image
        
        # correct_ALK_pos = 0
        # correct_ALK_neg = 0
        total_ALK_pos = 0
        total_ALK_neg = 0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size()[0]
            # print(labels.size())
            # print(labels[0].item())
            for j in range(batch_size):
                if labels[j].item() == 1:  # ALK+ 클래스
                    total_ALK_pos += 1
                    correct_ALK_pos += (predicted == labels).sum().item()
                else:  # ALK- 클래스
                    total_ALK_neg += 1
                    correct_ALK_neg += (predicted == labels).sum().item()

            correct += predicted.eq(labels).sum().item()
            if num_ % 10000 == 0:
                print(f" {i}th Iteration Loss : {loss.item()}")
            num_ += 1
    
    # ALK_pos_accuracy = 100. * correct_ALK_pos / total_ALK_pos
    # ALK_neg_accuracy = 100. * correct_ALK_neg / total_ALK_neg
    if total != 0:    
        print(f"Total ALK+ Image : {total_ALK_pos}  |   Correct : {correct_ALK_pos} |   Wrong : {total_ALK_pos-correct_ALK_pos} |   ALK+ Accuracy : {ALK_pos_accuracy}")
    #print("Test Loss: {:.3f} | Test Acc : {:.3f}".format(test_loss / i, 100. * correct/total), f"| Total Image, Correct : {total}, {correct}")

    # print(f"Total ALK- Image : {total_ALK_neg}  |   Correct : {correct_ALK_neg} |   Wrong : {total_ALK_neg-correct_ALK_neg} |   ALK+ Accuracy : {ALK_neg_accuracy}")