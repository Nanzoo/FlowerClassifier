import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import tensor
from torch import optim
import argparse
import json
import PIL
from PIL import Image
import time

parser = argparse.ArgumentParser()

parser.add_argument('--image', type=str, default='flowers/test/37/image_03734.jpg', help='input image path')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='trained model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='top k most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to actual names')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for inference')
parser.add_argument('--arch', type=str, default='vgg13', help='CNN architecture')

in_arg = parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_pil = Image.open(image)

    # define transforms
    preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    return img_tensor

def predict(image_path, model, topk=in_arg.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
 
    device = torch.device("cuda:0" if (torch.cuda.is_available() and in_arg.gpu == 'gpu') else "cpu")
    #if torch.cuda.is_available():
        #model.cuda()
    #model.to(device)
    with torch.no_grad():
        print(device)
        model.to(device)
       
        model.eval()
        image_tensor = (torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.cuda.FloatTensor) if in_arg.gpu == 'gpu' else                                       torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor))
       # image_tensor.to(device)
        forward2 = model.forward(image_tensor)
        putTop2 = torch.exp(forward2)
        tp2, tc2 = putTop2.topk(topk, dim=1)
        invertDict = {
                        model.class_to_idx[allTop] : allTop for allTop in model.class_to_idx
                     }
        if(in_arg.gpu == 'gpu'):
            topK_invDict = [
                            invertDict[cpuNump] for cpuNump in tc2.cpu().numpy()[0]
                           ]
        else:
            topK_invDict = [
                            invertDict[cpuNump] for cpuNump in tc2.cpu().numpy()[0]
                           ]
        
        if(in_arg.gpu == 'gpu'):
            topK_tp2 = tp2.cpu().numpy()[0]
        else:
            topK_tp2 = tp2.cpu().numpy()[0]
    
    return topK_tp2, topK_invDict

def main():
    checkpoint = torch.load(in_arg.checkpoint)
    _model = getattr(models, in_arg.arch)
    model = _model(pretrained=True)
    for allParam in model.parameters():
        allParam.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, 120), nn.ReLU(), nn.Dropout(0.4), nn.Linear(120, 90), nn.ReLU(), 
                                nn.Linear(90,70), nn.ReLU(), nn.Linear(70,102), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    image_pth = in_arg.image
    tp2, invDict = predict(image_pth, model)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    image_invDict = [cat_to_name[ctn] for ctn in invDict]
    print(image_invDict)
    print(tp2)

if __name__ == '__main__':
    main()
