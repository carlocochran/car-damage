from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from common import DAMAGE_TO_NUM_MAP

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None, path='./'):
        self.df = df
        self.df['class_d'] = self.df['class'].map(DAMAGE_TO_NUM_MAP)
        self.transform = transform
        self.target_transform = target_transform
        self.path = path

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.df.iloc[idx][0])
        image = read_image(img_path)/255
        label = self.df.iloc[idx][3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label