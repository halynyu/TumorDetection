import torchvision.models as models
import torch
import torch.nn as nn

def make_ResNet(args):

    epochs = args.epochs
    num_labels = args.labels

    model = models.resnet18(pretrained=True)
    new_conv1_weight = torch.randn((64, 3, 7, 7))
    model.conv1.weight = nn.Parameter(new_conv1_weight)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_labels)

    return model