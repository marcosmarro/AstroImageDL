import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    """One 3x3 convolution with ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetN2V(nn.Module):
    def __init__(self, in_ch=1, in_features=96, out_ch=1, depth=2):
        """UNet neural network from arXiv:1505.04597"""
        super().__init__()
        
        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        curr_ch  = in_ch
        features = in_features

        for _ in range(depth):
            self.down.append(DoubleConv(curr_ch, features))
            curr_ch   = features
            features *= 2

        self.bottleneck = DoubleConv(curr_ch, features)

        curr_ch = features

        for _ in range(depth):
            features = features // 2
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=curr_ch, out_channels=features, kernel_size=2, stride=2),
                DoubleConv(curr_ch, features)
            ))
            curr_ch = features

        self.final = nn.Conv2d(in_features, out_ch, kernel_size=1)

    def forward(self, x):
        
        skip = []

        for down in self.down:
            x = down(x)
            skip.append(x)
            x = F.max_pool2d(x, 2)

        x    = self.bottleneck(x)
        skip = skip[::-1]

        for idx, upblock in enumerate(self.up):
            x = upblock[0](x)   # upsample
            x = torch.cat((x, skip[idx]), dim=1)
            x = upblock[1](x)   # convolution
    
        x = self.final(x)

        return x
    

class UNetN2N(nn.Module):
    def __init__(self, in_ch=1, in_features=48, out_ch=1, depth=3):
        """UNet neural network from arXiv:1505.04597"""
        super().__init__()
        
        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        curr_ch  = in_ch
        features = in_features

        for i in range(depth):
            if i==0:
                self.down.append(DoubleConv(curr_ch, features))
                curr_ch = features
            else:
                self.down.append(SingleConv(curr_ch, curr_ch))
 
        self.bottleneck = SingleConv(curr_ch, curr_ch)

        features = curr_ch + in_features
    
        for i in range(depth - 1):
            if i==0:
                self.up.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=curr_ch, out_channels=curr_ch, kernel_size=2, stride=2),
                DoubleConv(features, features)
                ))

                curr_ch  = features 
                features += in_features 
                
            else:
                self.up.append(nn.Sequential(
                    nn.ConvTranspose2d(in_channels=curr_ch, out_channels=curr_ch, kernel_size=2, stride=2),
                    DoubleConv(features, curr_ch)
                ))

        self.finalup = nn.ConvTranspose2d(in_channels=curr_ch, out_channels=curr_ch, kernel_size=2, stride=2)

        features = curr_ch + 1

        self.final = nn.Sequential(
            SingleConv(features, 64),
            SingleConv(64, 32),
            SingleConv(32, out_ch))

    def forward(self, x):
        
        skip = [x]

        for down in self.down:
            x = down(x)
            skip.append(x)
            x = F.max_pool2d(x, 2)
        
        x    = self.bottleneck(x)
        skip = skip[::-1]
        
        for idx, upblock in enumerate(self.up):
            x = upblock[0](x)   # upsample
            x = torch.cat((x, skip[idx]), dim=1)
            x = upblock[1](x)   # convolution
      
        x = self.finalup(x)
        x = torch.cat((x, skip[-1]), dim=1)
        x = self.final(x)

        return x