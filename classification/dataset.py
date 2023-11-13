import os 
from PIL import Image 
import torch 
from torch.utils.data import Dataset 
import torchvision.transforms as transforms 


class TumorDataset(Dataset):
    def __init__(self, file_list, label, transform=None) -> None:
        super().__init__()
        self.file_list = file_list 
        self.label = label
        self.transform =transform 


    def __len__(self) ->None:
        return len(self.file_list) 
    
    def __getitem__(self, index) -> None:
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img) 
        # Image Path : PATCH/NEGATIVE/SVS_NAME/image ...  or   PATCH/POSITIVE/SVS_NAME/image ... 
        return img_transformed, self.label 
    
    
# define params 
class ImageTransform():
    def __init__(self) -> None:
        self.data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])
    
    def __call__(self, img):
        return self.data_transform(img)
        
