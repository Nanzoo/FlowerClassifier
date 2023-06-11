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

parser.add_argument('--data_dir', type=str, default='flowers', help='data root')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save the trained model to a checkpoint')
parser.add_argument('--arch', type=str, default='vgg13', help='CNN architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='hidden units')
parser.add_argument('--epochs', type=int, default=12, help='num of epochs')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for training')

in_arg = parser.parse_args()


def trainModel(arch, learnRate, epochs, trainData):
    _model = getattr(models, arch)
    model = _model(pretrained=True)
    for allParam in model.parameters():
        allParam.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, 120), nn.ReLU(), nn.Dropout(0.4), nn.Linear(120, 90), nn.ReLU(), 
                                nn.Linear(90,70), nn.ReLU(), nn.Linear(70,102), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and in_arg.gpu == 'gpu') else "cpu")
    print(device)
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # print(device)
    #if torch.cuda.is_available():
        #model.cuda()
    model.device = device
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learnRate)
    steps = 0
    print_every = 3
    test_loss = 0
    accuracy = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in trainData:
            #if(in_arg.gpu == 'gpu'):
            inputs = inputs.to(device)
            labels = labels.to(device)
           # else:
           # inputs = inputs.to('cpu')
           # labels = labels.to('cpu')
                
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
           # print("first loop")
            #if steps % print_every == 0:
             #   model.eval()
             #   test_loss = 0
              #  accuracy = 0
              #  for images, labels in dataloaders['trainLoader']:
                  #  print("second loop")
               #     images = images.to(device)
                #    labels = labels.to(device)
                #    logps = model(images)
                #    loss = criterion(logps, labels)
                 #   test_loss += loss.item()
                 #   ps = torch.exp(logps)
                 #   top_ps, top_class = ps.topk(1, dim=1)
                 #   equality = (top_class == labels.view(*top_class.shape))
                 #   accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            steps += 1
            print(f"Training Loss: {running_loss/len(trainData)}")

           # print(f"Training Loss: {running_loss/print_every:.3f}")
           # print(f"Test Loss: {test_loss/len(dataloaders['trainLoader']):.3f}")
           # print(f"Test Accuracy: {accuracy/len(dataloaders['trainLoader']):.3f}")
            #running_loss = 0
            #model.train()
    acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in trainData:
            print("Working")
            #if(in_arg.gpu == 'gpu'):
            images = images.to(device)
            labels = labels.to(device)
            #else:
            #images = images.to('cpu')
            #labels = labels.to('cpu')
            forward = model.forward(images)
            putTop = torch.exp(forward)
            tp, tc = putTop.topk(1, dim=1)
            equals = (tc == labels.view(*tc.shape))
            acc += torch.mean(equals.type(torch.FloatTensor)).item()
    print("The accuracy is: {:.3f}".format(100 * (acc/len(trainData))))
    return model

def saveCheck(imgData, mod):
    mod.class_to_idx = imgData.class_to_idx
    checkpoint = {'hidden_layer1':120, 'droupout':0.4, 'epochs':12, 'state_dict':mod.state_dict(), 'class_idx':mod.class_to_idx}
    torch.save(checkpoint, in_arg.save_dir)

def loadCheckPoint(mod):
    state_dict = torch.load(in_arg.save_dir)
    mod.load_state_dict(state_dict)
    print(state_dict.keys())


    
def main():
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms ={'trainTrans':transforms.Compose([transforms.RandomRotation(50), transforms.RandomResizedCrop(224), 
                                          transforms.RandomVerticalFlip(),transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                      'vaidTrans':transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                      'testTrans':transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder 
    image_datasets = {'trainImage': datasets.ImageFolder(train_dir, transform = data_transforms['trainTrans']),
                                        'validImage': datasets.ImageFolder(valid_dir, transform = data_transforms['vaidTrans']),
                                        'testImage': datasets.ImageFolder(test_dir, transform =  data_transforms['testTrans'])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'trainLoader':torch.utils.data.DataLoader(image_datasets['trainImage'], batch_size=32, shuffle=True),
                   'validLoader':torch.utils.data.DataLoader(image_datasets['validImage'], batch_size=32, shuffle=True),
                   'testLoader':torch.utils.data.DataLoader(image_datasets['testImage'], batch_size=32, shuffle=True)}
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model = trainModel(in_arg.arch, in_arg.learning_rate, in_arg.epochs, dataloaders['trainLoader'])
    saveCheck(image_datasets['trainImage'], model)
    
if __name__ == '__main__':
    main()
    
    

