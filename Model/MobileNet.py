import torch 
import torch.nn as nn 

class DepthwiseConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, stride) -> None:
        super().__init__()

        self.convolution = nn.Sequential(nn.Conv2d(in_channels=in_channel, 
                                                   out_channels=in_channel, 
                                                   kernel_size=3, 
                                                   stride = stride, 
                                                   padding=1, 
                                                   groups=in_channel),
                                         nn.GroupNorm(num_groups=32, num_channels=in_channel),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1),
                                         nn.GroupNorm(num_groups=32, num_channels=out_channel),
                                         nn.ReLU())
        
    def forward(self, x):
        return self.convolution(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.name = 'MobileNet'

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
                                        DepthwiseConvolution(in_channel=32, out_channel=32, stride=1))

        self.layer_2 = nn.Sequential(DepthwiseConvolution(in_channel=32, out_channel=64, stride=2),
                                     DepthwiseConvolution(in_channel=64, out_channel=64, stride=1))
        
        self.layer_3 = nn.Sequential(DepthwiseConvolution(in_channel=64, out_channel=128, stride=2),
                                     DepthwiseConvolution(in_channel=128, out_channel=128, stride=1))
        
        self.layer_4 = nn.Sequential(DepthwiseConvolution(in_channel=128, out_channel=256, stride=2),
                                     DepthwiseConvolution(in_channel=256, out_channel=256, stride=1),
                                     DepthwiseConvolution(in_channel=256, out_channel=256, stride=1),
                                     DepthwiseConvolution(in_channel=256, out_channel=256, stride=1),)
        
        self.layer_5 = nn.Sequential(DepthwiseConvolution(in_channel=256, out_channel=512, stride=2),
                                     DepthwiseConvolution(in_channel=512, out_channel=512, stride=1))
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(in_features=512, out_features=num_classes))
    
    def forward(self, x):
        x = self.init_layer(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x 

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ =='__main__':

    input = torch.randn((32,3,227,227))
    model = MobileNet(num_classes=5)
    output = model(input)

    print(output.shape)
    print("Model Parameters:", count_param(model))
