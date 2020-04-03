import os
from PIL import Image
from io import BytesIO
import base64
import torch
from torch.autograd import Variable as V
from torch.nn import functional as F
import torchvision.transforms as transforms

from gate.crop import Crop
from gate.model import Net


def load_model():
    model = Net()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = torch.load(
        dir_path + '/model.pth.tar', map_location='cpu')
    state_dict = {str.replace(k, 'module.', ''): v for k,
                  v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    return model


def b64_to_Image(b64_img):
    pic = BytesIO()
    img_string = BytesIO(base64.b64decode(b64_img))
    img = Image.open(img_string)
    img.save(pic, img.format, quality=100)
    pic.seek(0)
    return Image.open(pic)


# load the model
model = load_model()

# load the image transformer
tf = transforms.Compose([
    Crop((0, 0), (720, 720)),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def predict(image):
    # load the image
    img = b64_to_Image(image)

    input_img = V(tf(img).unsqueeze(0))

    # forward pass rooms
    output = model.forward(input_img)
    prob = output.data.squeeze().item()
    print('output', output, prob)

    return [
        {'tag': 'open', 'confidence': prob},
        {'tag': 'closed', 'confidence': 1 - prob}
    ]
