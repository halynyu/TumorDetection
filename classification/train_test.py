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

    train_total = 0
    train_correct = 0

    train_total_ALK_pos = 0
    train_correct_ALK_pos = 0
    train_total_ALK_neg = 0
    train_correct_ALK_neg = 0

    num = 0

    save_path = os.path.join(save_path, f'{weights[0]}_{weights[1]}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    running_loss = 0.0

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            train_total += labels.size()[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 각 레이블에 대한 예측 확인
            _, predicted = torch.max(outputs, 1)  # 각 샘플에 대한 예측값을 얻어옴

            for j in range(labels.size()[0]):
                current_label = labels[j].item()
                current_predicted = predicted[j].item()
            
                # print(f"Sample {j + 1} - True Label: {current_label}, Predicted Label: {current_predicted}")

                if current_label == 1:
                    train_total_ALK_pos += 1
                    train_correct_ALK_pos += (current_predicted == current_label)
                else:
                    train_total_ALK_neg += 1
                    train_correct_ALK_neg += (current_predicted == current_label)
                
                train_correct += (current_predicted == current_label)

            num += 1

        if train_total_ALK_pos != 0:
            train_ALK_pos_accuracy = 100. * train_correct_ALK_pos / train_total_ALK_pos
            print(f"\n\nALK+ train accuracy : {train_ALK_pos_accuracy}")
        if train_total_ALK_neg != 0:
            train_ALK_neg_accuracy = 100. * train_correct_ALK_neg / train_total_ALK_neg
            print(f"ALK- train accuracy : {train_ALK_neg_accuracy}")

        



            # Validation
            model.eval()
            correct_predictions = 0
            total_samples = 0
            
            validation_total = 0
            validation_correct = 0

            validation_total_ALK_pos = 0
            validation_correct_ALK_pos = 0
            validation_total_ALK_neg = 0
            validation_correct_ALK_neg = 0


            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    predicted_labels = torch.argmax(torch.sigmoid(outputs), dim=1)

                    correct_predictions += (predicted_labels == labels).sum().item()
                    total_samples += len(labels)

                validation_accuracy = correct_predictions / total_samples
                validation_loss = sum(criterion(torch.sigmoid(model(x.to(device))), torch.eye(2, device=device)[y.to(device)]).item() for x, y in valid_dataloader)

                for k in range(labels.size()[0]):
                    current_label = labels[k].item()
                    current_predicted = predicted_labels[k].item()

                    if current_label == 1:
                        validation_total_ALK_pos += 1
                        validation_correct_ALK_pos += (current_predicted == current_label)
                    else:
                        validation_total_ALK_neg += 1
                        validation_correct_ALK_neg += (current_predicted == current_label)

            if validation_total_ALK_pos != 0:
                validation_ALK_pos_accuracy = 100. * validation_correct_ALK_pos / validation_total_ALK_pos
                print(f"ALK+ validation accuracy : {validation_ALK_pos_accuracy}")

            if validation_total_ALK_neg != 0:
                validation_ALK_neg_accuracy = 100. * validation_correct_ALK_neg / validation_total_ALK_neg
                print(f"ALK- validation accuracy : {validation_ALK_neg_accuracy}")


                if BEST_LOSS > validation_loss:
                    BEST_LOSS = validation_loss
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, os.path.join(save_path,f'epoch_{epoch}_all.tar'))


        end_time = time.time()
        print(f" EPOCH : {epoch + 1} / {epochs} | Validation Loss per Epoch : {validation_loss / len(valid_dataloader)} | Validation Accuracy: {validation_accuracy * 100:.2f}% | Training Time per epoch : {end_time - start_time}s \n",end='\n')

def eval(model, dataloader, criterion, device, batch_size, image_save, dataset):

    len_dataset = len(dataset)

    model.to(device)
    model.eval()
    criterion.to(device)

    min_loss = 1e9
    min_loss_image = None
    test_loss = 0
    correct = 0
    batch_count = 0
    total = 0

    save_path = 'evaluation_image/20231109/'
    if image_save:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

    num_ = 0

    total_ALK_pos = 0
    correct_ALK_pos = 0
    total_ALK_neg = 0
    correct_ALK_neg = 0

    with torch.no_grad():
        import torchvision
        import numpy as np
        from PIL import Image

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            total += labels.size()[0]
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            if labels.item() == 1:
                total_ALK_pos += 1
                correct_ALK_pos += predicted.eq(labels).sum().item()
            else:
                total_ALK_neg += 1
                correct_ALK_neg += predicted.eq(labels).sum().item()

            correct += predicted.eq(labels).sum().item()

    if total_ALK_pos != 0:
        ALK_pos_accuracy = 100. * correct_ALK_pos / total_ALK_pos
        print(f"ALK+ accuracy: {ALK_pos_accuracy}")

    if total_ALK_neg != 0:
        ALK_neg_accuracy = 100. * correct_ALK_neg / total_ALK_neg
        print(f"ALK- accuracy: {ALK_neg_accuracy}")

    print(" Test Acc: {:.3f}".format(100. * correct / total), f"| Total Image, Correct: {total}, {correct}")
