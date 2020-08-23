import torch
import torch.nn as nn


def weights_init(m):    
    if isinstance(m, nn.Conv2d):        
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        #m.weight.data.normal_(0, 0.01)
        #m.bias.data = torch.ones(m.bias.data.size())


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
        
        self.max_pool_part = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )
        
        self.avg_pool_part = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()                  
        )      
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 64)
        )              
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=6//2)
        )
    
    def CBAM_attention(self, out):
        max_pool_part_out = self.shared_mlp(self.max_pool_part(out))
        avg_pool_part_out = self.shared_mlp(self.avg_pool_part(out))
        
        #channel_attention = nn.Sigmoid()(max_pool_part_out + avg_pool_part_out)        
        channel_attention = nn.ReLU()(max_pool_part_out + avg_pool_part_out)        
        channel_attention_out = out * channel_attention.view(channel_attention.shape[0], channel_attention.shape[1], 1, 1)        
        
        channel_wise_max_pooling = torch.max(channel_attention_out, 1)[0].unsqueeze(1)
        channel_wise_mean_pooling = torch.mean(channel_attention_out, 1).unsqueeze(1)
        
        channel_wise_pool = torch.cat([channel_wise_max_pooling, channel_wise_mean_pooling], dim=1)
        #spatial_attention_out = nn.Sigmoid()(self.conv7(channel_wise_pool))
        spatial_attention_out = nn.ReLU()(self.conv7(channel_wise_pool))
        
        attention_out = channel_attention_out * spatial_attention_out
        
        '''
        attention_out2 = channel_attention_out * spatial_attention_out.view(spatial_attention_out.shape[0], 1, spatial_attention_out.shape[2], spatial_attention_out.shape[3])
        print(channel_attention_out.shape)
        print(spatial_attention_out.shape)
        print(attention_out.shape)
        print(attention_out2.shape)
        print(torch.all(torch.eq(attention_out, attention_out2)))
        assert 1 == 2        
        '''
        
        return attention_out  
                           
    def forward(self, x):
        out = self.layer1(x)
        #out = self.CBAM_attention(out)
        
        out = self.layer2(out)
        #out = self.CBAM_attention(out)
        
        out = self.layer3(out)
        #out = self.CBAM_attention(out)
        
        out = self.layer4(out)
        #out = self.CBAM_attention(out)
        
        return out


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
        
        self.max_pool_part = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten()
        )
        
        self.avg_pool_part = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()                  
        )      
        
        self.shared_mlp64 = nn.Sequential(
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 64)
        )
        
        self.shared_mlp128 = nn.Sequential(
            nn.Linear(128, 128 // 16),
            nn.ReLU(),
            nn.Linear(128 // 16, 128)
        )              
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=6//2)
        )
        
    def CBAM_attention(self, out, channel_size):
        max_pool_part_out = None
        avg_pool_part_out = None
        if channel_size == 128:
            max_pool_part_out = self.shared_mlp128(self.max_pool_part(out))
            avg_pool_part_out = self.shared_mlp128(self.avg_pool_part(out))
        elif channel_size == 64:
            max_pool_part_out = self.shared_mlp64(self.max_pool_part(out))
            avg_pool_part_out = self.shared_mlp64(self.avg_pool_part(out))
        
        assert max_pool_part_out is not None
        assert avg_pool_part_out is not None
        
        channel_attention = nn.ReLU()(max_pool_part_out + avg_pool_part_out)        
        channel_attention_out = out * channel_attention.view(channel_attention.shape[0], channel_attention.shape[1], 1, 1)        
        
        channel_wise_max_pooling = torch.max(channel_attention_out, 1)[0].unsqueeze(1)
        channel_wise_mean_pooling = torch.mean(channel_attention_out, 1).unsqueeze(1)
        
        channel_wise_pool = torch.cat([channel_wise_max_pooling, channel_wise_mean_pooling], dim=1)
        spatial_attention_out = nn.ReLU()(self.conv7(channel_wise_pool))

        attention_out = channel_attention_out * spatial_attention_out
       
        return attention_out  
    
    def forward(self, x):        
        out = self.CBAM_attention(x, 128)           
        out = self.layer1(out)     
        #out = self.layer1(x)     
        out = self.CBAM_attention(out, 64)
        out = self.layer2(out)                
        out = self.CBAM_attention(out, 64)
        out = self.layer3(out)
        
        return out
