import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.io import read_image
from PIL import Image
import numpy as np
from common import NUM_TO_DAMAGE_MAP
import base64
import io

BASE_DIR = './'
MODEL_NAME = BASE_DIR + 'car_damage_0.001_0.3_1000.pth'
IMAGE_NAME = BASE_DIR + 'preprocessed/image/0.jpeg'
NUM_CLASSES = 8
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
transform = T.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))

def predict(encoded_string):
    net = torchvision.models.resnet50()
    net.fc = nn.Linear(2048, NUM_CLASSES)
    net.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device('cpu')))

    base64_decoded = base64.b64decode(encoded_string)
    img = Image.open(io.BytesIO(base64_decoded))
    image = torch.tensor(np.array(img))/255
    image = torch.movedim(image, 2, 0)
    image = transform(image)
    #save_image(image, './resized_img.jpg')

    # image = read_image(IMAGE_NAME)/255

    image = torch.unsqueeze(image, dim=0)
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)

    return NUM_TO_DAMAGE_MAP[predicted.item()]

# enc_str = None
# with open(IMAGE_NAME, 'rb') as image_file:
#     enc_str = base64.b64encode(image_file.read()).decode('ascii')
#
# print(enc_str)
# print(predict(enc_str))
