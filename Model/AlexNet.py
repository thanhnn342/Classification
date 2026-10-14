import torch 
import torch.nn as nn 

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.name = "AlexNet"

        self.layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
                                     nn.GroupNorm(num_groups=16, num_channels=64),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                                     nn.GroupNorm(num_groups=16, num_channels=128),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer_3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=16, num_channels=256),
                                     nn.ReLU(),

                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=16, num_channels=256),
                                     nn.ReLU(),

                                     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=16, num_channels=256),
                                     nn.ReLU(),

                                     nn.MaxPool2d(kernel_size=3, stride=2)
                                     )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = self.mlp(x)
        return x

def count_param(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ =='__main__':

    input = torch.randn((1,3,227,227))
    model = AlexNet(num_classes=6)
    output = model(input)

    print(output.shape)
    print("Model Parameters:", count_param(model))