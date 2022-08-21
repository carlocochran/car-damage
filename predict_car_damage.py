import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image

BASE_DIR = './'
MODEL_NAME = BASE_DIR + 'car_damage_0.01_0.3.pth'
IMAGE_NAME = BASE_DIR + 'preprocessed/image/146.jpeg'
NUM_CLASSES = 8

di = {0: 'unknown', 1: 'door_scratch', 2: 'head_lamp', 3: 'glass_shatter', 4: 'tail_lamp', 5: 'bumper_dent',
      6: 'door_dent', 7: 'bumper_scratch'}

net = torchvision.models.resnet50()
net.fc = nn.Linear(2048, NUM_CLASSES)
net.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device('cpu')))

image = read_image(IMAGE_NAME)/255
image = torch.unsqueeze(image, dim=0)
outputs = net(image)
_, predicted = torch.max(outputs, 1)

print(predicted)
