import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
    
import helper
model = models.resnet18(pretrained=True)  
  
from collections import OrderedDict

#Custom classifier
fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(512,8)),
                                ('ReLU', nn.ReLU()),
                                ('fc2', nn.Linear(8,2)),
                                ('output', nn.LogSoftmax(dim = 1))]))

#Updating model with custom classifier
model.fc = fc


#Loading saved model parameters
state_dict = torch.load('ResNet18CatDog.pth')
model.load_state_dict(state_dict)

#Pipeline
test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                     (0.229, 0.224, 0.225))])

test_root = 'Custom_Cat_Dog_data'
test_data = datasets.ImageFolder(test_root, transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)




import time

t0 = time.time()
#Evaluating
accuracy = 0
# model.cuda()
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        # inputs, labels = inputs.cuda(), labels.cuda()
        logps = model.forward(inputs)
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
t1 = time.time()

total = t1-t0       
        
        
print('Test Accuracy:', 100*accuracy/len(testloader),'%')
print('It took',total,'seconds to run on CPU')