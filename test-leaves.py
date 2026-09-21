import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torch.utils.data import Dataset
import os

labels_dataframe = pd.read_csv('../data/classify-leaves/train.csv')
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v : k for k, v in class_to_num.items()}

class Leaves_dataset(Dataset):
    def __init__(self,csv_path,img_path,model,valid_ratio=0.2):
        self.csv_path = csv_path
        self.img_path = img_path
        self.model = model
        # 读取csv文件
        self.data=pd.read_csv(csv_path)
        self.data_len=len(self.data)
        self.train_len=int(self.data_len*(1-valid_ratio))
        if self.model=='train':
            self.transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ])
    def __getitem__(self, index):
        if self.model=='train':
            real_idx=index
        elif self.model=='valid':
            real_idx=index+self.train_len
        else:   # test
            real_idx=index

        img_name=self.data.iloc[real_idx,0]
        img_path=os.path.join(self.img_path,img_name)
        image=Image.open(img_path).convert('RGB')
        image=self.transform(image)
        if self.model=='test':
            return image
        else:
            label=class_to_num[self.data.iloc[real_idx,1]]
            return image,label
    def __len__(self):
        if self.model=='train':
            return self.train_len
        elif self.model=='valid':
            return self.data_len-self.train_len
        else:
            return self.data_len

train_path = '../data/classify-leaves/train.csv'
test_path = '../data/classify-leaves/test.csv'
img_path='../data/classify-leaves/images'

train_dataset=Leaves_dataset(train_path,img_path,'train')
valid_dataset=Leaves_dataset(train_path,img_path,'valid')
test_dataset=Leaves_dataset(test_path,img_path,'test')
print(train_dataset)
print(test_dataset)
print(test_dataset)