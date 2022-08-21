import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from common import NUM_TO_DAMAGE_MAP

BASE_DIR = './'
MODEL_NAME = BASE_DIR + 'car_damage_0.01_0.3.pth'
IMAGE_NAME = BASE_DIR + 'preprocessed/image/158.jpeg'
NUM_CLASSES = 8

net = torchvision.models.resnet50()
net.fc = nn.Linear(2048, NUM_CLASSES)
net.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device('cpu')))

image = read_image(IMAGE_NAME)/255
image = torch.unsqueeze(image, dim=0)
outputs = net(image)
_, predicted = torch.max(outputs, 1)

print(NUM_TO_DAMAGE_MAP[predicted.item()])
