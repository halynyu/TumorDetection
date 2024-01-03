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



# LUAC : LUAC_train_dataloader, LUAC_valid_dataloader / others : train_dataloader, valid_dataloader
def train(model, LUAC_dataloader, dataloader, valid_dataloader, epochs, save_path, device, LUAC_weights, weights):
    BEST_LOSS = 1e11
    model.to(device)

    train_total = 0
    train_correct = 0
    train_total_pos = 0
    train_total_neg = 0
    train_correct_pos = 0
    train_correct_neg = 0


    save_path = os.path.join(save_path, f"{LUAC_weights[0]}_{LUAC_weights[1]}/{weights[0]}_{weights[1]}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        # LUAC
        LUAC_pos, LUAC_neg = LUAC_weights
        train_one_epoch(model, LUAC_dataloader, LUAC_weights, device, train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg)
        

        # Others
        pos, neg = weights
        train_one_epoch(model, dataloader, weights, device, train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg)

    if train_total_pos != 0:
        train_pos_accuracy = 100. * train_correct_pos / train_total_pos
        print(f"\n\nALK+ train accuracy : {train_pos_accuracy}")
    
    if train_total_neg != 0:
        train_neg_accuracy = 100. * train_correct_neg / train_total_neg
        print(f"ALK- train accuracy : {train_neg_accuracy}")


    # validation 및 ROC 곡선 plot
    # validation_loss, validation_accuracy = validate(model, valid_dataloader, criterion, device)
    
    



def train_one_epoch(model, dataloader, weights, device, total, correct, pos, neg, correct_pos, correct_neg):
    model.train()

    pos, neg = weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        print(f"Input device : {inputs.device}")
        print(f"Target device : {labels.device}")
        print(device)
        loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device).index_select(0, labels))
        total += labels.size()[0]
        loss.backward()
        optimizer.step()
    

        _, predicted = torch.max(outputs, 1)

        for j in range(labels.size()[0]):
            current_label = labels[j].item()
            current_prediction = predicted[j].item()

            if current_label == 1:
                pos += 1
                correct_pos += (current_predicted == current_label)
            else:
                neg += 1
                correct_neg += (current_predicted == current_label)
            
            correct += (current_predicted == current_labels)

    return total, correct, pos, neg, correct_pos, correct_neg
        




def validate(model, dataloader, weights, device):
    model.eval()

    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)

            pos_mask = (labels == 1)
            neg_mask = (labels == 0)

            correct_pos += (predicted[pos_mask] == labels[pos_mask]).sum().item()
            correct_neg += (predicted[neg_mask] == labels[neg_mask]).sum().item()

            total_pos += alk_pos_mask.sum().item()
            total_neg += alk_neg_mask.sum().item()

    accuracy_pos = correct_pos / total_pos if total_pos > 0 else -1
    accuracy_neg = correct_neg / total_neg if total_neg > 0 else -1

    if accuracy_pos != -1:
        printf("ALK+ accuracy : {accuracy_pos * 100:.2f}%")
    if accuracy_neg != -1:
        printf("ALK- accuracy : {accuracy_neg * 100:.2f}%")

