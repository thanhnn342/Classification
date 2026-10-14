import torch 
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self,in_channel, intermediate_channels, out_channels):
        super().__init__()

        self.paral1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=1),
                                    nn.GroupNorm(num_groups=8, num_channels=out_channels[0]),
                                    nn.ReLU())

        self.paral2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channels[0], kernel_size=1),
                                    nn.GroupNorm(num_groups=8, num_channels=intermediate_channels[0]),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=intermediate_channels[0], out_channels=out_channels[1], kernel_size=3, padding=1),
                                    nn.GroupNorm(num_groups=8, num_channels=out_channels[1]),
                                    nn.ReLU(),)
        
        self.paral3 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channels[1], kernel_size=1),
                                    nn.GroupNorm(num_groups=8, num_channels=intermediate_channels[1]),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=intermediate_channels[1], out_channels=out_channels[2], kernel_size=5, padding=2),
                                    nn.GroupNorm(num_groups=8, num_channels=out_channels[2]),
                                    nn.ReLU())
        
        self.paral4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(in_channels=in_channel, out_channels=out_channels[3], kernel_size=1),
                                    nn.GroupNorm(num_groups=8, num_channels=out_channels[3]),
                                    nn.ReLU(),)

    def forward(self, x):
        x1 = self.paral1(x)
        x2 = self.paral2(x)
        x3 = self.paral3(x)
        x4 = self.paral4(x)

        output = torch.concat([x1, x2 , x3, x4], dim=1)
        return output

class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.name = 'GoogLeNetV3'

        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                     nn.GroupNorm(num_groups=32, num_channels=64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=32, num_channels=128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception1 = nn.Sequential(InceptionModule(in_channel=128, intermediate_channels=[96, 16], out_channels=[64, 128, 32, 32]),
                                        InceptionModule(in_channel=256, intermediate_channels=[128, 32], out_channels=[128, 192, 96, 64]),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception2 = nn.Sequential(InceptionModule(in_channel=480, intermediate_channels=[96, 16], out_channels=[192, 208, 48, 64]),
                                        InceptionModule(in_channel=512, intermediate_channels=[112, 24], out_channels=[160, 224, 64, 64]),
                                        InceptionModule(in_channel=512, intermediate_channels=[112, 24], out_channels=[160, 224, 64, 64]),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception3 = nn.Sequential(InceptionModule(in_channel=512, intermediate_channels=[192, 48], out_channels=[384, 384, 128, 128]),
                                        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(in_features=1024, out_features=num_classes))
    
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.inception1(x)

        x = self.inception2(x)

        x = self.inception3(x)

        x = self.avg_pool(x)

        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x 
    

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ =='__main__':

    input = torch.randn((1,3,227,227))
    model = GoogLeNet(num_classes=6)
    output = model(input)

    print(output.shape)
    print("Model Parameters:", count_param(model))