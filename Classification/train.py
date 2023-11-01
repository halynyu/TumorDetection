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
from dataset import TumorDataset, ImageTransform
from tqdm import tqdm
import time
import wandb


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
    parser.add_argument('--model_save_path', type=str, default='model_save/patch_32_AB_DE/',
                        help="Where to Save model ")
    parser.add_argument('--pretrained', type=bool,  default=False,
                        help="Using Pretrained or not")
    parser.add_argument('--loss_image_save', type=bool, default =False,
                        help="Sampling Image getting High Loss ")
    args = parser.parse_args()

    return args

def make_ResNet50(args):
    num_labels = args.labels

    model = models.resnet50(pretrained=True)
    new_conv1_weight = torch.randn(((643, )))

def make_ResNet(args):

    epochs = args.epochs
    num_labels = args.labels

    model = models.resnet18(pretrained=True)
    new_conv1_weight = torch.randn((64, 3, 7, 7))
    model.conv1.weight = nn.Parameter(new_conv1_weight)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_labels)

    return model

"""
ㅇ Patch

    ㅇ NEGATIVE
        - POS_DATASET1
            -- image_0
            -- image_1
            -- image 2
            .....

        - POS_DATASET2
            -- image_0
            -- image_1
            -- image 2
            .....

        - ...


    ㅇ POSITIVE
        - NEG_DATASET1
            -- image_0
            -- image_1
            -- image 2
            .....

        - NEG_DATASET2
            -- image_0
            -- image_1
            -- image 2
            .....

        - ...

"""

def ABDE_CF_create_dataloader(NEGATIVE_PATH, POSITIVE_PATH):
    negative_svs_patch_folders_train = os.listdir(NEGATIVE_PATH)[:2] # NEG_DATASET1, NEG_DATASET2, NEG_DATASET3, ..
    positive_svs_patch_folders_train = os.listdir(POSITIVE_PATH)[:2] # POS_DATASET1, POS_DATASET2, POS_DATASET3, ..

    negative_svs_patch_folders_valid = os.listdir(NEGATIVE_PATH)[2] # NEG_DATASET1, NEG_DATASET2, NEG_DATASET3, ..
    positive_svs_patch_folders_valid = os.listdir(POSITIVE_PATH)[2] # POS_DATASET1, POS_DATASET2, POS_DATASET3, ..

    print(f"NEGATIVE TRAIN FOLDER : {negative_svs_patch_folders_train} | POSITIVE TRAIN FOLDER : {positive_svs_patch_folders_train} ")
    print(f"NEGATIVE VALID FOLDER : {negative_svs_patch_folders_valid} | POSITIVE VALID FOLRDER : {positive_svs_patch_folders_valid}")

    patch_path = []

    number_of_negative, number_of_positive = 0, 0

    print("\n\nDataset PATH Stored in Folder .....\n ")
    for svs_patch_folder in negative_svs_patch_folders_train:
        current_folder = os.path.join(NEGATIVE_PATH, svs_patch_folder)
        print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
        number_of_negative += len(os.listdir(current_folder))
        for image_name in os.listdir(current_folder):
            patch_path.append(os.path.join(current_folder, image_name))


    for svs_patch_folder in positive_svs_patch_folders_train:
        current_folder = os.path.join(POSITIVE_PATH, svs_patch_folder)
        print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
        number_of_positive += len(os.listdir(current_folder))
        for image_name in os.listdir(current_folder):
            patch_path.append(os.path.join(current_folder, image_name))

    # negative_list_train = patch_path[:-number_of_positive]
    # positive_list_train = patch_path[number_of_negative:]
    # print(type(negative_list), type(positive_list))

    def shuffle_datasets_path(data_path):
        import random
        random.seed(77)
        random.shuffle(data_path)

    # patch_path = []
    # shuffle_datasets_path(list(negative_list_train))

    # for n in negative_list_train:
    #     patch_path.append(n)
    # for p in positive_list_train:
    #     patch_path.append(p)
    # shuffle_datasets_path(patch_path)

    shuffle_datasets_path(patch_path) # PATCH/NEGATIVE/SF18_65507/image_name.jpg, PATCH/POSITIVE/SS18_23306/image.jpg ...

    print("Save Dataset List : {} images".format(len(patch_path)))

    train_image_size = int(len(patch_path))


    print(f"Train Image size : {train_image_size}")

    MEAN = (0.485, 0.456, 0.456)
    STD = (0.229, 0.224, 0.225)
    print(patch_path[:100])
    train_dataset = TumorDataset(patch_path, transform = ImageTransform(mean= MEAN, std =STD))
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)



    ''' validation DATA '''
    patch_path = []
    number_of_negative, number_of_positive = 0, 0

    print("\n\nDataset PATH Stored in Folder .....\n ")
    svs_patch_folder = negative_svs_patch_folders_valid
    current_folder = os.path.join(NEGATIVE_PATH, svs_patch_folder)
    print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
    number_of_negative += len(os.listdir(current_folder))
    for image_name in os.listdir(current_folder):
        patch_path.append(os.path.join(current_folder, image_name))


    svs_patch_folder = positive_svs_patch_folders_valid
    current_folder = os.path.join(POSITIVE_PATH, svs_patch_folder)
    print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
    number_of_positive += len(os.listdir(current_folder))
    for image_name in os.listdir(current_folder):
        patch_path.append(os.path.join(current_folder, image_name))

    negative_list = patch_path[:-number_of_positive]
    positive_list = patch_path[number_of_negative:]
    # print(type(negative_list), type(positive_list))

    print(len(negative_list), len(positive_list))
    def shuffle_datasets_path(data_path):
        import random
        random.seed(77)
        random.shuffle(data_path)

    patch_path = []
    shuffle_datasets_path(list(negative_list))

    for n in negative_list[:number_of_positive]:
        patch_path.append(n)
    for p in positive_list:
        patch_path.append(p)


    shuffle_datasets_path(patch_path) # PATCH/NEGATIVE/SF18_65507/image_name.jpg, PATCH/POSITIVE/SS18_23306/image.jpg ...

    print("Save Dataset List : {} images".format(len(patch_path)))

    valid_dataset = TumorDataset(patch_path, transform= ImageTransform(mean=MEAN, std=STD))
    valid_dataloader = DataLoader(valid_dataset, batch_size= args.batch_size, shuffle=False)


    return train_dataloader, valid_dataloader


def create_dataloader(NEGATIVE_PATH, POSITIVE_PATH):

    negative_svs_patch_folders = os.listdir(NEGATIVE_PATH) # NEG_DATASET1, NEG_DATASET2, NEG_DATASET3, ..
    positive_svs_patch_folders = os.listdir(POSITIVE_PATH) # POS_DATASET1, POS_DATASET2, POS_DATASET3, ..

    print(positive_svs_patch_folders, negative_svs_patch_folders)


    # NEG_DATASET1/image_0, NEG_DATASET2/image_1, ...
    patch_path = []

    number_of_negative, number_of_positive = 0, 0

    print("\n\nDataset PATH Stored in Folder .....\n ")
    for svs_patch_folder in negative_svs_patch_folders:
        current_folder = os.path.join(NEGATIVE_PATH, svs_patch_folder)
        print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
        number_of_negative += len(os.listdir(current_folder))
        for image_name in os.listdir(current_folder):
            patch_path.append(os.path.join(current_folder, image_name))


    for svs_patch_folder in positive_svs_patch_folders:
        current_folder = os.path.join(POSITIVE_PATH, svs_patch_folder)
        print(f" Number of {current_folder} Images : {len(os.listdir(current_folder))}")
        number_of_positive += len(os.listdir(current_folder))
        for image_name in os.listdir(current_folder):
            patch_path.append(os.path.join(current_folder, image_name))

    print(f" Total Number Of Images : {number_of_negative + number_of_positive}" )
    print(f"Using Dataset Nums : {2 * number_of_positive}")

    negative_list = patch_path[:-number_of_positive]
    positive_list = patch_path[number_of_negative:]
    # print(type(negative_list), type(positive_list))

    def shuffle_datasets_path(data_path):
        import random
        random.seed(77)
        random.shuffle(data_path)

    patch_path = []
    shuffle_datasets_path(list(negative_list))

    for n in negative_list:
        patch_path.append(n)
    for p in positive_list:
        patch_path.append(p)


    shuffle_datasets_path(patch_path) # PATCH/NEGATIVE/SF18_65507/image_name.jpg, PATCH/POSITIVE/SS18_23306/image.jpg ...

    print("Save Dataset List : {} images".format(len(patch_path)))

    train_image_size = int(len(patch_path) * 0.7)
    valid_image_size = int(len(patch_path) * 0.2)
    test_image_size = int(len(patch_path) * 0.1)

    print(f"Train Image size : {train_image_size} | Valid Image size : {valid_image_size} | Test Image size : {test_image_size}")
    MEAN = (0.485, 0.456, 0.456)
    STD = (0.229, 0.224, 0.225)

    train_dataset = TumorDataset(patch_path[:train_image_size], transform = ImageTransform(mean= MEAN, std =STD))
    valid_dataset = TumorDataset(patch_path[train_image_size: train_image_size + valid_image_size], transform= ImageTransform(mean= MEAN, std =STD))
    test_dataset = TumorDataset(patch_path[: -test_image_size])

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader


def train(model, dataloader, criterion, optimizer, epochs, save_path, device):
    wandb.watch(model, criterion,log="all", log_freq=10)
    BEST_LOSS = 1e11
    model.to(device)
    model.train()
    print(device)
    criterion.to(device)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        running_loss = 0.0
        for i, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels)
            # print(f" Batch Input Shape(input, label) : {inputs.shape} {labels.shape} ")
            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs)
            # print(f"Batch Output Shape : {outputs.shape}")
            # _, predicted = outputs.max(1)
            # print(f"predicted Shape : {predicted.shape}")
            # print(predicted)
            # labels = labels.float()
            loss= criterion(torch.sigmoid(outputs), torch.eye(2, device=device)[labels])
            loss.backward()
            optimizer.step()

            '''
            # print(f" Batch Input Shape(input, label) : {inputs.shape} {labels.shape} ")
            # print(outputs)
            # print(f"Batch Output Shape : {outputs.shape}")
            # print(f"predicted Shape : {predicted.shape}")
            # print(predicted)
            '''

            running_loss += loss.item()
            print(f"{i}th Iteration Loss : {loss.item()} ")
        running_loss /= float(i)

        if BEST_LOSS > running_loss:
            BEST_LOSS = running_loss
            torch.save(model.state_dict(), os.path.join(save_path,f'model_state_dict_{epoch}.pt'))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(save_path,f'epoch_{epoch}_all.tar'))

        wandb.log({"Training LOSS":running_loss})
        end_time = time.time()
        print(f" EPOCH : {epoch + 1} / {epochs} | Loss per Epoch : {running_loss} | Training Time per epoch : {end_time - start_time}s ",end='\n')


def eval(model, dataloader, criterion, device='cpu', image_save=False):
    model.to(device)
    model.eval()

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

    with torch.no_grad():
        import torchvision
        import numpy as np
        from PIL import Image


        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += labels.size()[0]
            test_loss += loss.item()
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()

            print(f" {i}th Iteration Loss : {loss.item()}")

    print("Test Loss: {:.3f} | Test Acc : {:.3f}".format(test_loss / i, 100. * correct/total), f"| Total Image, Correct : {total}, {correct}")

def make_plots(DATAPATH):
    from PIL import Image
    import matplotlib.pyplot as plt
    image_paths = os.listdir(DATAPATH)
    images = [Image.open(os.path.join(DATAPATH,path)) for path in image_paths]

    row, col = 4, 8
    fig, axes = plt.subplots(row, col, figsize=(20, 10))

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for i in range(row):
        for j in range(col):
            image_index = i * col + j
            label = image_paths[image_index].split(".")[0][-1]
            if label == '0' :
                label = 'ALK-'
            else :
                label = 'ALK+'

            axes[i, j].imshow(images[image_index])
            axes[i, j].axis('off')
            # axes[i, j].set_title(, fontsize=10)
            axes[i, j].set_title(label, fontsize=10)
            axes[i, j].set_aspect('equal')

    fig.suptitle("incorrect max loss Images ", fontsize=40)
    plt.savefig(DATAPATH+'/plot.png')

def eval_and_saveimage(model,testloader,num_image, device):
    ALK_p, ALK_n = 0, 0
    model.to(device)
    import numpy as np
    model.eval()
    correct_images = []
    incorrect_images = []
    losses = []
    correct, incorrect = 0,0
    with torch.no_grad():
        for i,(images, labels) in enumerate(testloader) :
            if( i% 100 == 0):
                print(f"{i}th Iteration.....")
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
            _, predicted = outputs.max(1)

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct += 1
                    if len(correct_images) <100000:
                        correct_images.append((loss[i].item(), images[i], labels[i]))

                else:
                    incorrect +=1
                    if len(incorrect_images)< 100000:
                        incorrect_images.append((loss[i].item(), images[i], labels[i]))
            # print(f'predicted Shape : {predicted.shape} | loss Shape: {loss.shape}')
            # print(f"Correct Image List Length : {len(correct_images)} | Incorrect Image List Lenght : {len(incorrect_images)}")

    print(f"Accuracy : {len(correct_images)}/{len(correct_images) + len(incorrect_images)}")
    correct_images.sort(key=lambda x: x[0])
    incorrect_images.sort(key=lambda x:x[0], reverse=True)

    for i in range(num_image):
        img = correct_images[i][1]
        label = correct_images[i][2]
        if label : # POs = 1
            label = 1
        else :
            label = 0
        save_path = os.path.join("LOSS_DATA/correct_min_loss_0515", f"image{i}_label_{label}.jpg")
        torchvision.utils.save_image(img, save_path)

    for i in range(num_image):
        img = incorrect_images[i][1]
        label = incorrect_images[i][2]
        if label :
            label = 1
            ALK_p +=1
        else :
            label = 0
            ALK_n +=1
        save_path = os.path.join("LOSS_DATA\incorrect_max_loss_0515", f"image{i}_label_{label}.jpg")
        torchvision.utils.save_image(img, save_path)

"""1 KB미만의 데이터는 일단삭제해보자. """
def delete_trash_data(path):
    pass

def make_Accuracy_image(model):
    model.eval()
    sigmoid = nn.Sigmoid()

    from PIL import Image, ImageDraw
    import re


    JPEG_PATH = ["JPEG_FOLDER/SS18-19871_N.jpg", "JPEG_FOLDER/SS22-65597_P.jpg"]
    NEGATIVE_PATH = "Patch/NEGATIVE_32_0513/SS18-198"
    POSITIVE_PATH = "Patch/POSITIVE_32_0513/SS22-655"
    PATH_ = [NEGATIVE_PATH, POSITIVE_PATH]

    jpeg_image = Image.open(JPEG_PATH[0])
    width, height = jpeg_image.size

    white_image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(white_image)
    with torch.no_grad():
        image_path = NEGATIVE_PATH
        current_path = image_path
        for image_ in os.listdir(image_path):
            i_p = os.path.join(current_path,image_)
            image = Image.open(i_p)
            pattern = r'(?<=_)\d+'
            numbers = re.findall(pattern,image_)
            w, h = numbers
            w, h = map(int, (w,h))


            transform = transforms.ToTensor()
            image = transform(image).to(device)
            # print(image.shape)
            # GPU 로 넣어야함
            image = torch.unsqueeze(image, dim = 0)
            output = model(image)
            probs = sigmoid(output)[0][0].item()
            print(f"width : {w} | Height : {h} | Prob : {probs}")
            red = int(probs * 255)  # ALK- label = 0
            blue = int((1-probs) * 255) # ALK+ 일경우 Label = 1

            draw.rectangle((w,h, w+2, h+2), fill = (red, 0, blue))
            # Image 크기에 해당하는 백지를 만들고, 그에 해당하는 부분 w,h에 (red,0,blue를 넣자.)

    white_image.save(f'SigmoidAttention/negative_weighted_1.6.jpg')

    jpeg_image = Image.open(JPEG_PATH[1])
    width, height = jpeg_image.size

    white_image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(white_image)
    with torch.no_grad():
        image_path = POSITIVE_PATH
        current_path = image_path
        for image_ in os.listdir(image_path):
            i_p = os.path.join(current_path,image_)
            image = Image.open(i_p)
            pattern = r'(?<=_)\d+'
            numbers = re.findall(pattern,image_)
            w, h = numbers
            w, h = map(int, (w,h))

            transform = transforms.ToTensor()
            image = transform(image).to(device)
            # print(image.shape)
            # GPU 로 넣어야함
            image = torch.unsqueeze(image, dim = 0)
            output = model(image)
            probs = sigmoid(output)[0][0].item()
            print(f"width : {w} | Height : {h} | Prob : {probs}")
            red = int(probs * 255)  # ALK+
            blue = int((1-probs) * 255) # ALK- 일경우

            draw.rectangle((w,h, w+2, h+2), fill = (red, 0, blue))
            # Image 크기에 해당하는 백지를 만들고, 그에 해당하는 부분 w,h에 (red,0,blue를 넣자.)

    white_image.save(f'SigmoidAttention/positive_weighted_1.6.jpg')


"""가중치 클래스"""
class WeightedBCELoss(nn.Module):
    def __init__(self, weight) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        bce_loss =nn.BCELoss(reduction='none')(input, target)
        WeightedBCELoss = torch.mean(bce_loss * self.weight)
        return WeightedBCELoss



if __name__ == '__main__':

    # Define model, criterion, optimizer, batch_size, epoch
    args = get_args()
    print(args.epochs)

    wandb.init(
        project="Training-ABEF-Patch32-level=0-BinaryTumorClassification",

        config= {
            "learning_rate": args.learning_rate,
            "Architecture" : "Custom ResNet-18",
            "dataset" : "Tumor",
            "epochs" : args.epochs,
        }
    )
    model = make_ResNet(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("**************** Model Architecture **********************************\n")
    print(model)
    print("********************************************************************\n\n")

    model.to(device)
    # if not os.path.exist(args.model_save_path):
    #     os.mkdirs(args.model_save_path)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path, exist_ok=True)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.6, 1]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    if args.pretrained :
        checkpoint = torch.load('2023_05_25_train_weight_1.6_1/epoch_19_all.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    print(device)

    # Create Dataset (ALK+ vs ALK-)

    # train_dataloader, valid_dataloader, test_dataloader = create_dataloader("Patch/NEGATIVE/", "Patch/POSITIVE/") # 이후에 데이터셋이 만들어지면 넣>기


    # print(f" Batch Size : {args.batch_size} | learing_rate : {args.learning_rate} | Epoch : {args.epochs}")


    # Train
    # print("\nStart Training!!        \n")
    # train(model, train_dataloader, criterion, optimizer, args.epochs, args.model_save_path, device)
    # print("\n Finish Training Model !!")

    # Valid
    # print("\nEvaluate Model !!       ")
    # eval(model, valid_dataloader, criterion, device, args.loss_image_save)
    # eval_and_saveimage(model,valid_dataloader,32, device)
    # print("Finish Evaluation Model !!")



    # make_plots("LOSS_DATA/correct_min_loss_0413/")
    # make_plots("LOSS_DATA/incorrect_max_loss_0413/")


    # """ ABDE / CF Dataset으로 훈련하는 파트 """
    # train_dataloader, valid_dataloader = ABDE_CF_create_dataloader("Patch/NEGATIVE_32_0513/", "Patch/POSITIVE_32_0513/")
    # print("\nStart Training!!        \n")
    # train(model, train_dataloader, criterion, optimizer, args.epochs, args.model_save_path, device)
    # print("\n Finish Training Model !!")

    """ Evaluation Part"""
    # eval_and_saveimage(model,valid_dataloader,32, device)

    """ E,F 이미지에 대해서 Attention Score를 만들기"""

    make_Accuracy_image(model)