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
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train(model, LUAC_dataloader, dataloader, valid_dataloader, LUAC_criterion, LUAC_optimizer, criterion, optimizer, epochs, save_path, device, LUAC_weights, weights):
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

    auc_values = []

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        # LUAC train
        train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg = train_one_epoch(model, LUAC_dataloader, LUAC_weights, LUAC_criterion, LUAC_optimizer, device, train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg)
        print(train_total)

        # 나머지 train
        train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg = train_one_epoch(model, dataloader, weights, criterion, optimizer, device, train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg)

        if train_total_pos != 0:
            train_pos_acc = 100. * train_correct_pos / train_total_pos
            print(f"\n\nALK+ train Accuracy : {train_pos_acc}")
        else:
            print("There is no ALK+ train data")

        if train_total_neg != 0:
            train_neg_acc = 100. * train_correct_neg / train_total_neg
            print(f"ALK- train Accuracy : {train_neg_acc}")
        else:
            print("There is no ALK- train data")

    
        # validate
        valid_loss, valid_auc = validate(model, valid_dataloader, weights, auc_values, device)
        print(f"Validation AUC after epoch {epoch + 1} : {valid_auc}")
        auc_values.append(valid_auc)


    max_auc_epoch = plot_and_save_auc(auc_values, epochs, save_path)

    for epoch in range(epochs):
        retrain_to_maximum_AUC(model, LUAC_dataloader, LUAC_criterion, LUAC_optimizer, device)
    
    for epoch in range(max_auc_epoch):
        retrain_to_maximum_AUC(model, dataloader, criterion, optimizer, device)

    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }, os.path.join(save_path, f"epoch_{max_auc_epoch}_all.tar"))


        


def train_one_epoch(model, dataloader, weights, criterion, optimizer, device, total, correct, pos, neg, correct_pos, correct_neg):
    model.train()
    criterion.to(device)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device).index_select(0, labels))
        total += labels.size()[0]
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        for j in range(labels.size()[0]):
            current_labels = labels[j].item()
            current_prediction = predicted[j].item()

            if current_labels == 1:
                pos += 1
                correct_pos += (current_prediction == current_labels)
            else:
                neg += 1
                correct_neg += (current_prediction == current_labels)

            correct += (current_prediction == current_labels)

        return total, correct, pos, neg, correct_pos, correct_neg





def validate(model, dataloader, weights, auc_values, device):
    model.eval()

    train_loss = 0
    all_labels = []
    all_scores = []
    pos, neg = weights

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
    criterion.to(device)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels.to(device)])

            train_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(torch.sigmoid(outputs).cpu().numpy())
        
    average_loss = train_loss / len(dataloader)

    all_labels = torch.tensor(all_labels, device='cpu').numpy()
    all_scores = torch.tensor(all_scores, device='cpu').numpy()
    all_scores = [x[0] for x in all_scores]


    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auc_value = roc_auc_score(all_labels, all_scores)

    # ROC 곡선 그리기
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="b", lw=2, label=f"AUC = {auc_value:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


    return average_loss, auc_value



def criterion_optimizer(model, weights):
    pos, neg = weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer



def plot_and_save_auc(auc_values, epochs, save_path):
    
    # AUC값을 dataframe으로 변환
    data = {'Epoch': range(1, epochs + 1), 'AUC': auc_values}
    df = pd.DataFrame(data)

    # 데이터프레임을 csv파일로 저장
    csv_path = os.path.join(save_path, "auc_values.csv")
    df.to_csv(csv_path, index=False)

    # 그래프로 plot
    plt.plot(df['Epoch'], df['AUC'], marker='o', linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('AUC Value')
    plt.title('AUC Value over Epochs')
    
    plot_path = os.path.join(save_path, 'auc_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # 최대 AUC값을 가진 epoch 찾기
    max_auc_epoch = df[df['Epoch'] >= 50]['AUC'].idxmax()
    max_auc_value = df['AUC'].max()
    print(f"Maximum AUC value at epoch {max_auc_epoch}: max_auc_value")

    return max_auc_epoch



def retrain_to_maximum_AUC(model, dataloader, criterion, optimizer, device):
    model.train()
    criterion.to(device)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device).index_select(0, labels))
        loss.backward()
        optimizer.step()