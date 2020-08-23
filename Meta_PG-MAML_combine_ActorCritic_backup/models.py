import torch
import torch.nn as nn
from torch.distributions import Categorical

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
            nn.MaxPool2d(2)           
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out


class ActorCritic(nn.Module):
    """
    docstring for RelationNetwork
    """    
    def __init__(self, input_size, hidden_size, class_num):
        super(ActorCritic, self).__init__()
        self.class_num = class_num
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
        
        self.actor = nn.Sequential(
            Flatten(),
            nn.Linear(input_size * 3 * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )               
        
        self.critic = nn.Sequential(
            Flatten(),
            nn.Linear(input_size * 3 * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )   
        
    
    def forward(self, x):  
        out = self.layer1(x)     
        out = self.layer2(out) 
        
        logit = self.actor(out)
        logit = logit.view(-1, self.class_num)
        prob = nn.Softmax(dim=-1)(logit)
        dist = Categorical(prob)        
        
        value = self.critic(out).view(-1, self.class_num)
        value = nn.Softmax(dim=-1)(value)
        
        return dist, value
    