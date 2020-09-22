import os
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)

if not torch.cuda.is_available():
    model.load_state_dict(torch.load('./model_newdatasetfinal.pth',  map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('./model_newdatasetfinal.pth'))
    
model.eval();
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize((224,224)), 
                                transforms.ToTensor()])

def openEyeCheck(imgPath=None):
    '''Eye image classifer(open/close) 
    Args:
        imgPath (str): full path to an image
    Return:
        out (int): 0 if eye is closed else 1
    '''
    if imgPath is None:
        print('Please, pass a full path to an image')
    img = Image.open(imgPath)
    img = transform(img).float()
    img = torch.cat(3*[img])
    img = normalize(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
    return out.numpy()
