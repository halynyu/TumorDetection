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

def train(model, LUAC_dataloader, dataloader, valid_dataloader, LUAC_valid_dataloader, YS_valid_dataloader, TCGA_valid_dataloader, SSSF_valid_dataloader, LUAC_criterion, LUAC_optimizer, criterion, optimizer, epochs, save_path, device, LUAC_weights, weights):
    BEST_LOSS = 1e11
    model.to(device)


    train_total = 0
    train_correct = 0
    train_total_pos = 0
    train_total_neg = 0
    train_correct_pos = 0
    train_correct_neg = 0

    LUAC_train_total = 0
    LUAC_train_correct = 0
    LUAC_train_total_pos = 0
    LUAC_train_total_neg = 0
    LUAC_train_correct_pos = 0
    LUAC_train_correct_neg = 0

    save_path = os.path.join(save_path, f"{LUAC_weights[0]}_{LUAC_weights[1]}/{weights[0]}_{weights[1]}")


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    acc_values = []

    train_LUAC_pos = []
    train_LUAC_neg = []
    train_pos = []
    train_neg = []
    val_LUAC_pos = []
    val_LUAC_neg = []
    val_YS_pos = []
    val_TCGA_pos = []
    val_SSSF_pos = []
    val_SSSF_neg = []

    for epoch in tqdm(range(epochs)):
        # 여기는 epoch

        LUAC_train_total, LUAC_train_correct, LUAC_train_total_pos, LUAC_train_total_neg, LUAC_train_correct_pos, LUAC_train_correct_neg = train_one_epoch(model, LUAC_dataloader, LUAC_weights, LUAC_criterion, LUAC_optimizer, device, LUAC_train_total, LUAC_train_correct, LUAC_train_total_pos, LUAC_train_total_neg, LUAC_train_correct_pos, LUAC_train_correct_neg)

        # 나머지 train
        train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg = train_one_epoch(model, dataloader, weights, criterion, optimizer, device, train_total, train_correct, train_total_pos, train_total_neg, train_correct_pos, train_correct_neg)


        if (epoch+1) % 50 == 0:
            if LUAC_train_total_pos != 0:
                LUAC_train_pos_acc = 100. * LUAC_train_correct_pos / LUAC_train_total_pos
            else:
                LUAC_train_pos_acc = "No LUAC_ALK+ train data"

            if LUAC_train_total_neg != 0:
                LUAC_train_neg_acc = 100. * LUAC_train_correct_neg / LUAC_train_total_neg
            else:
                LUAC_train_neg_Acc = "No LUAC_ALK- train data"

            if train_total_pos != 0:
                train_pos_acc = 100. * train_correct_pos / train_total_pos
            else:
                train_pos_acc = "No SSSF/TCGA/YS ALK+ train data"

            if train_total_neg != 0:
                train_neg_acc = 100. * train_correct_neg / train_total_neg
            else:
                train_neg_acc = "No SSSF/TCGA/YS ALK- train data"

            print(f"{epoch + 1} Train Accuracy\nLUAC    |   ALK+ : {LUAC_train_pos_acc}   |   ALK- : {LUAC_train_neg_acc}\nOthers   |   ALK+ : {train_pos_acc}  |   ALK- : {train_neg_acc}")
            train_LUAC_pos.append(LUAC_train_pos_acc)
            train_LUAC_neg.append(LUAC_train_neg_acc)
            train_pos.append(train_pos_acc)
            train_neg.append(train_neg_acc)

        LUAC_valid_acc, LUAC_valid_pos_acc, LUAC_valid_neg_acc = validate(model, LUAC_valid_dataloader, weights, LUAC_criterion, device)
        YS_valid_acc, YS_valid_pos_acc, YS_valid_neg_acc = validate(model, YS_valid_dataloader, weights, criterion, device)
        TCGA_valid_acc, TCGA_valid_pos_acc, TCGA_valid_neg_acc = validate(model, TCGA_valid_dataloader, weights, criterion, device)
        SSSF_valid_acc, SSSF_valid_pos_acc, SSSF_valid_neg_acc = validate(model, SSSF_valid_dataloader, weights, criterion, device)
        valid_acc, valid_pos_acc, valid_neg_acc, acc_values, validation_loss = validate_all(model, valid_dataloader, weights, acc_values, criterion, device)


    
        if (epoch+1) % 50 == 0:
            print(f"{epoch + 1} Validation Accuracy")
            print(f"LUAC    |   ALK+ : {LUAC_valid_pos_acc} |   ALK- : {LUAC_valid_neg_acc} |   total accuracy : {LUAC_valid_acc}")
            print(f"YS      |   ALK+ : {YS_valid_pos_acc} |   ALK- : {YS_valid_neg_acc} |   total accuracy : {YS_valid_acc}")
            print(f"TCGA    |   ALK+ : {TCGA_valid_pos_acc} |   ALK- : {TCGA_valid_neg_acc} |   total accuracy : {TCGA_valid_acc}")
            print(f"SSSF    |   ALK+ : {SSSF_valid_pos_acc} |   ALK- : {SSSF_valid_neg_acc} |   total accuracy : {SSSF_valid_acc}")
            print(f"\nTotal   |   ALK+ : {valid_pos_acc}      |   ALK- : {valid_neg_acc}      |   total accuracy : {valid_acc}")

            
            val_LUAC_pos.append(LUAC_valid_pos_acc)
            val_LUAC_neg.append(LUAC_valid_neg_acc)
            val_YS_pos.append(YS_valid_pos_acc)
            val_TCGA_pos.append(TCGA_valid_pos_acc)
            val_SSSF_pos.append(SSSF_valid_pos_acc)
            val_SSSF_neg.append(SSSF_valid_neg_acc)

        if BEST_LOSS > validation_loss:
            BEST_LOSS = validation_loss
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_path,f'epoch_{epoch}_all.tar'))

        

def eval(model, dataloader, criterion, optimizer, device, batch_size, model_path):

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.to(device)
    model.eval()
    criterion.to(device)

    min_loss = 1e9
    test_loss = 0
    correct = 0
    batch_count = 0
    total = 0

    total_pos = 0
    correct_pos = 0
    total_neg = 0
    correct_neg = 0

    save_path = "evaluation_image/20232306/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            total += labels.size()[0]
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            for j in range(labels.size()[0]):
                current_labels = labels[j].item()
                current_prediction = predicted[j].item()

                if current_labels == 1:
                    total_pos += 1
                    correct_pos += (current_prediction == current_labels)
                else:
                    total_neg += 1
                    correct_neg += (current_prediction == current_labels)

                correct += (current_prediction == current_labels)
            
        if total_pos == 0:
            pos_acc = "No ALK+ test data"
        else:
            pos_acc = 100. * correct_pos / total_pos

        if total_neg == 0:
            neg_acc = "No ALK- test data"
        else:
            neg_acc = 100. * correct_neg / total_neg

        total_acc = 100. * correct / total
        
    return total_pos, correct_pos, total_neg, correct_neg, pos_acc, neg_acc, total_acc


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


def validate(model, dataloader, weights, criterion,device):
    model.eval()

    correct_predictions = 0
    total_samples = 0

    total = 0
    correct = 0

    total_pos = 0
    correct_pos = 0
    total_neg = 0
    correct_neg = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_labels = torch.argmax(torch.sigmoid(outputs), dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += len(labels)

        accuracy = 100. * correct_predictions / total_samples 
        loss = sum(criterion(torch.sigmoid(model(x.to(device))), torch.eye(2, device=device)[y.to(device)]).item() for x, y in dataloader)

        for k in range(labels.size()[0]):
            current_label = labels[k].item()
            current_predicted = predicted_labels[k].item()

            if current_label == 1:
                total_pos += 1
                correct_pos += (current_predicted == current_label)
            else:
                total_neg += 1
                correct_neg += (current_predicted == current_label)

    if total_pos != 0:
        pos_acc = 100. * correct_pos / total_pos
    else:
        pos_acc = "No ALK+ validation data"

    if total_neg != 0:
        neg_acc = 100. * correct_neg / total_neg
    else:
        neg_acc = "No ALK- validation data"


    return accuracy, pos_acc, neg_acc


def validate_all(model, dataloader, weights, acc_values, criterion, device):
    model.eval()

    correct_predictions = 0
    total_samples = 0

    total = 0
    correct = 0

    total_pos = 0
    correct_pos = 0
    total_neg = 0
    correct_neg = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted_labels = torch.argmax(torch.sigmoid(outputs), dim=1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += len(labels)

        accuracy = 100. * correct_predictions / total_samples
        loss = sum(criterion(torch.sigmoid(model(x.to(device))), torch.eye(2, device=device)[y.to(device)]).item() for x, y in dataloader)

        for k in range(labels.size()[0]):
            current_label = labels[k].item()
            current_predicted = predicted_labels[k].item()

            if current_label == 1:
                total_pos += 1
                correct_pos += (current_predicted == current_label)
            else:
                total_neg += 1
                correct_neg += (current_predicted == current_label)

    if total_pos != 0:
        pos_acc = 100. * correct_pos / total_pos
    else:
        pos_acc = "No ALK+ validation data"

    if total_neg != 0:
        neg_acc = 100. * correct_neg / total_neg
    else:
        neg_acc = "No ALK- validation data"

    acc_values.append(accuracy)

    return accuracy, pos_acc, neg_acc, acc_values, loss

def criterion_optimizer(model, weights):
    pos, neg = weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos, neg]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return criterion, optimizer

def plot_and_save_acc(acc_values, epochs, save_path, LUAC_weights, weights):

    write_accuracy_csv(save_path, "total_validation_accuracy", acc_values_a=acc_values, acc_values_b=None, dual=False)

    print(f"Data has been written to {save_path}")    

    x_values = [i * 50 for i in range(1, len(acc_values)//50 + 1)]
    y_values = acc_values[49::50]

    # 그래프로 plot
    plt.plot(x_values, y_values, marker="o", linestyle="-")
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Plot Every 50 Epochs')

    plt.text(0.5, -0.15, f'LUAC Weight: {LUAC_weights}, Others Weight: {weights}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_path, 'accuracy_plot.png')
    plt.savefig(save_path)

    print('Plot saved as accuracy_plot.png')

def write_accuracy_csv(save_path, path, acc_values_a=None, acc_values_b=None, dual=True):
    csv_path = os.path.join(save_path, path + ".csv")

    if dual == True:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'Positive Accuracy', 'Negative Accuracy'])

            for epoch, (positive_acc, negative_acc) in enumerate(zip(acc_values_a, acc_values_b), start=1):
                csv_writer.writerow([epoch, positive_acc, negative_acc])

    if dual == False:
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'Positive Accuracy'])

            for epoch, accuracy in enumerate(acc_values_a, start=1):
                csv_writer.writerow([epoch, accuracy])

