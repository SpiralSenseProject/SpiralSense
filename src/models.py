#######################################################
# This file stores all the models used in the project.#
#######################################################

import torch
from torchvision.models import resnet50

#VGG16 model
class VGG16(torch.nn.Module):

        def __init__(self, num_classes):
            super().__init__()
            
            self.block_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_3 = torch.nn.Sequential(        
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            
            self.block_4 = torch.nn.Sequential(   
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_5 = torch.nn.Sequential(
                    
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),    
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))             
            )
                
            height, width = 3, 3 
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512*height*width, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, num_classes),
            )
                
            for m in self.modules():
                if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.detach().zero_()
                        
            self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))
            
            
        def forward(self, x):

            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.block_5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1) # flatten
            
            logits = self.classifier(x)
            #probas = F.softmax(logits, dim=1)

            return logits     

        
# ResNet18 model
