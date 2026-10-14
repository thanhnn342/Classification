import torch 
import torch.nn as nn 

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channel),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channel),)
        
        self.skip_net = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1) if in_channel != out_channel else nn.Identity()
        self.activation = nn.ReLU()
    def forward(self, x):
        h = x 
        x = self.net(x)
        h = self.skip_net(h)
        x = x + h
        x = self.activation(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name = "ResNet"
        self.first_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3),
                                         nn.GroupNorm(num_groups=4,num_channels=16),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.resblk_layer_1 = nn.Sequential(ResBlock(in_channel=16, out_channel=16),
                                            ResBlock(in_channel=16, out_channel=16),
                                            nn.MaxPool2d(kernel_size=3, stride=2))
        self.resblk_layer_2 = nn.Sequential(ResBlock(in_channel=16, out_channel=32),
                                            ResBlock(in_channel=32, out_channel=32),
                                            nn.MaxPool2d(kernel_size=3, stride=2))
        self.resblk_layer_3 = nn.Sequential(ResBlock(in_channel=32, out_channel=64),
                                            ResBlock(in_channel=64, out_channel=64),
                                            nn.MaxPool2d(kernel_size=3, stride=2),)
        self.resblk_layer_4 = nn.Sequential(ResBlock(in_channel=64, out_channel=128),
                                            ResBlock(in_channel=128, out_channel=128),
                                           )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Dropout(0.3),
                                  nn.Linear(in_features=128, out_features=256),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(256, out_features=num_classes))

    def forward(self, x):
        x = self.first_layer(x)
        x = self.resblk_layer_1(x)
        x = self.resblk_layer_2(x)
        x = self.resblk_layer_3(x)
        x = self.resblk_layer_4(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x 
    

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ =='__main__':

    input = torch.randn((16,3,227,227))
    model = ResNet(num_classes=5)
    output = model(input)

    print(output.shape)
    print("Model Parameters:", count_param(model))