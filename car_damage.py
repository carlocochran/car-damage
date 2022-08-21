import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd

np.random.seed(42)

DATA_PATH = './preprocessed/'
TRAINING_PCT = 0.7
VALIDATION_PCT = 0.15
TEST_PCT = 0.15
BATCH_SIZE = 128
# LEARNING_RATE = 0.001
LEARNING_RATE_RANGE = [0.01, 0.001]
# MOMENTUM = 0.9
MOMENTUM_RANGE = [0.5, 0.9]
EPOCH = 2
NUM_CLASSES = 8

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        di = {'unknown': 0, 'door_scratch': 1, 'head_lamp': 2, 'glass_shatter': 3, 'tail_lamp': 4, 'bumper_dent': 5,
              'door_dent': 6, 'bumper_scratch': 7}
        self.df['class_d'] = self.df['class'].map(di)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        img_path = os.path.join(DATA_PATH, self.df.iloc[idx][0])
        image = read_image(img_path)/255
        label = self.df.iloc[idx][3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

df = pd.read_csv(DATA_PATH + 'index.csv')

train, validate, test = np.split(df.sample(frac=1),
                                 [int(TRAINING_PCT*len(df)), int((TRAINING_PCT+VALIDATION_PCT)*len(df))])

for LEARNING_RATE in LEARNING_RATE_RANGE:
    for MOMENTUM in MOMENTUM_RANGE:
        print('Learning Rate: {:f}, Momentum: {:f}'.format(LEARNING_RATE, MOMENTUM))
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(512, NUM_CLASSES)
        if torch.cuda.is_available():
            net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

        trainset = CustomImageDataset(train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

        validateset = CustomImageDataset(validate)
        validateloader = torch.utils.data.DataLoader(validateset, batch_size=BATCH_SIZE, shuffle=False)

        for epoch in range(EPOCH):
            training_running_loss = 0.0
            validation_acc = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                training_loss = criterion(outputs, labels)
                training_loss.backward()
                optimizer.step()
                training_running_loss += training_loss.item()
                print(f'[{epoch + 1}, {i + 1:5d}] training loss: {training_running_loss}')

            validation_running_loss = 0.0
            for i, data in enumerate(validateloader, 0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                net.eval()
                outputs = net(inputs)
                validation_loss = criterion(outputs, labels)
                validation_running_loss += validation_loss.item()
                print(f'[{epoch + 1}, {i + 1:5d}] validation loss: {validation_running_loss}')
                _, predicted = torch.max(outputs, 1)
                validation_acc += torch.sum(predicted == labels)

            print('Epoch: {:d}, Learning Rate: {:f}, Momentum: {:f}, Accuracy: {:f}, Training Loss: {:f}, Validation Loss: {:f}'
                  .format(epoch+1, LEARNING_RATE, MOMENTUM, (validation_acc/len(validate.index)).item(),
                  training_running_loss, validation_running_loss))
