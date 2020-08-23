import torch
import torch.nn as nn

import numpy as np
CLASS_NUM = 5  
SAMPLE_NUM_PER_CLASS = 5  


def weights_init(m):    
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, shape):
        return shape.view(shape.size(0), -1)


class CNNEncoder(nn.Module):
    """
    Docstring for ClassName
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.Dropout2d()
        )
        
        self.layer4 = nn.Sequential(            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out()        
        
        self.linear = nn.Sequential(            
            Flatten(),
            nn.Linear(conv_out_size, 10),            
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.Softmax(dim=-1)
        )
        
    def _get_conv_out(self):       
        temp = torch.zeros([CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 84, 84]) 
        o1 = self.layer1(temp)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)
        
        conv_out = int(np.prod(o4.size()[1:]))
        # * e.g. conv_out : [25, 40000]
        return conv_out
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out
    
    def get_linear(self, out):                
        linear = self.linear(out)
        return linear        
        

class RelationNetwork(nn.Module):
    """
    docstring for RelationNetwork
    """    
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            Flatten(),
            nn.Linear(input_size * 3 * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        return out
