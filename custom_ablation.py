import torch
import torch.nn as nn
import torch.nn.functional as F

# --- SEBlock ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- DepthwiseSeparableConv ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

# --- InitialBlock (Shared by all models) ---
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x):
        conv = self.conv(x)
        pool = self.pool(x)
        out = torch.cat([conv, pool], dim=1)
        out = self.bn(out)
        out = self.prelu(out)
        return out

# --- Bottleneck for ENet ---
class BottleneckENet(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None, 
                 bottleneck_type='regular', dropout_prob=0.1, dilation=1, asymmetric=False):
        super(BottleneckENet, self).__init__()
        internal_channels = out_channels // 4 if internal_channels is None else internal_channels
        
        layers = []
        layers.append(nn.Conv2d(in_channels, internal_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))
        
        if bottleneck_type == 'down':
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=2, padding=1))
        elif asymmetric:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), 
                                   padding=(2, 0)))
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), 
                                   padding=(0, 2)))
        else:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=3, 
                                   stride=1, padding=dilation, dilation=dilation))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))
        
        layers.append(nn.Conv2d(internal_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.main = nn.Sequential(*layers)
        
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels or bottleneck_type == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=2 if bottleneck_type == 'down' else 1),
                nn.BatchNorm2d(out_channels)
            )
        
        self.prelu = nn.PReLU(out_channels)
    
    def forward(self, x):
        main = self.main(x)
        main = self.dropout(main)
        shortcut = self.shortcut(x)
        out = main + shortcut
        out = self.prelu(out)
        return out

# --- Bottleneck for CustomENet ---
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None,
                 bottleneck_type='regular', dropout_prob=0.1, dilation=1, asymmetric=False):
        super(Bottleneck, self).__init__()
        internal_channels = out_channels // 4 if internal_channels is None else internal_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, internal_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))

        if bottleneck_type == 'down':
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1))
        elif asymmetric:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0)))
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2)))
            layers.append(nn.BatchNorm2d(internal_channels))
            layers.append(nn.PReLU(internal_channels))
        else:
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3,
                                                stride=1, padding=dilation, dilation=dilation))
        layers.append(nn.Conv2d(internal_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(SEBlock(out_channels))

        self.main = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if in_channels != out_channels or bottleneck_type == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if bottleneck_type == 'down' else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        main = self.main(x)
        main = self.dropout(main)
        shortcut = self.shortcut(x)
        out = main + shortcut
        out = self.prelu(out)
        return out

# --- Bottleneck for No SEBlock ---
class BottleneckNoSE(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None,
                 bottleneck_type='regular', dropout_prob=0.1, dilation=1, asymmetric=False):
        super(BottleneckNoSE, self).__init__()
        internal_channels = out_channels // 4 if internal_channels is None else internal_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, internal_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))

        if bottleneck_type == 'down':
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1))
        elif asymmetric:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0)))
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2)))
            layers.append(nn.BatchNorm2d(internal_channels))
            layers.append(nn.PReLU(internal_channels))
        else:
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3,
                                                stride=1, padding=dilation, dilation=dilation))
        layers.append(nn.Conv2d(internal_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))

        self.main = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if in_channels != out_channels or bottleneck_type == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if bottleneck_type == 'down' else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        main = self.main(x)
        main = self.dropout(main)
        shortcut = self.shortcut(x)
        out = main + shortcut
        out = self.prelu(out)
        return out

# --- Bottleneck for No DepthwiseSeparableConv ---
class BottleneckNoDW(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None,
                 bottleneck_type='regular', dropout_prob=0.1, dilation=1, asymmetric=False):
        super(BottleneckNoDW, self).__init__()
        internal_channels = out_channels // 4 if internal_channels is None else internal_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, internal_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))

        if bottleneck_type == 'down':
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1))
        elif asymmetric:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0)))
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2)))
            layers.append(nn.BatchNorm2d(internal_channels))
            layers.append(nn.PReLU(internal_channels))
        else:
            layers.append(nn.Conv2d(internal_channels, internal_channels, kernel_size=3,
                                    stride=1, padding=dilation, dilation=dilation))
        layers.append(nn.Conv2d(internal_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(SEBlock(out_channels))

        self.main = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if in_channels != out_channels or bottleneck_type == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if bottleneck_type == 'down' else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        main = self.main(x)
        main = self.dropout(main)
        shortcut = self.shortcut(x)
        out = main + shortcut
        out = self.prelu(out)
        return out

# --- Bottleneck for No Asymmetric Conv ---
class BottleneckNoAsym(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels=None,
                 bottleneck_type='regular', dropout_prob=0.1, dilation=1, asymmetric=False):
        super(BottleneckNoAsym, self).__init__()
        internal_channels = out_channels // 4 if internal_channels is None else internal_channels

        layers = []
        layers.append(nn.Conv2d(in_channels, internal_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(internal_channels))
        layers.append(nn.PReLU(internal_channels))

        if bottleneck_type == 'down':
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3, stride=2, padding=1))
        else:
            layers.append(DepthwiseSeparableConv(internal_channels, internal_channels, kernel_size=3,
                                                stride=1, padding=dilation, dilation=dilation))
        layers.append(nn.Conv2d(internal_channels, out_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(SEBlock(out_channels))

        self.main = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if in_channels != out_channels or bottleneck_type == 'down':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if bottleneck_type == 'down' else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        main = self.main(x)
        main = self.dropout(main)
        shortcut = self.shortcut(x)
        out = main + shortcut
        out = self.prelu(out)
        return out

# --- ENet (Base Model) ---
class ENet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(ENet, self).__init__()
        
        self.initial = InitialBlock(in_channels, 16)
        
        self.stage1 = nn.Sequential(
            BottleneckENet(16, 64, bottleneck_type='down', dropout_prob=0.01),
            BottleneckENet(64, 64, dropout_prob=0.01)
        )
        
        self.stage2 = nn.Sequential(
            BottleneckENet(64, 128, bottleneck_type='down', dropout_prob=0.1),
            BottleneckENet(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckENet(128, 128, asymmetric=True, dropout_prob=0.1)
        )
        
        self.stage3 = nn.Sequential(
            BottleneckENet(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckENet(128, 128, asymmetric=True, dropout_prob=0.1),
            BottleneckENet(128, 128, dilation=4, dropout_prob=0.1)
        )
        
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            BottleneckENet(64, 64, dropout_prob=0.1)
        )
        
        self.stage5 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            BottleneckENet(16, 16, dropout_prob=0.1)
        )
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final(x)
        x = self.upsample(x)
        return x

# --- CustomENet ---
class CustomENet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CustomENet, self).__init__()

        self.initial = InitialBlock(in_channels, 16)

        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, bottleneck_type='down', dropout_prob=0.01),
            Bottleneck(64, 64, dropout_prob=0.01)
        )

        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, bottleneck_type='down', dropout_prob=0.1),
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1)
        )

        self.stage3 = nn.Sequential(
            Bottleneck(128, 128, dilation=2, dropout_prob=0.1),
            Bottleneck(128, 128, asymmetric=True, dropout_prob=0.1),
            Bottleneck(128, 128, dilation=4, dropout_prob=0.1)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            Bottleneck(64, 64, dropout_prob=0.1)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            Bottleneck(16, 16, dropout_prob=0.1)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final(x)
        x = F.interpolate(x, size=(720, 1280), mode='bilinear', align_corners=True)
        return x

# --- CustomENet without SEBlock ---
class CustomENetNoSE(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CustomENetNoSE, self).__init__()

        self.initial = InitialBlock(in_channels, 16)

        self.stage1 = nn.Sequential(
            BottleneckNoSE(16, 64, bottleneck_type='down', dropout_prob=0.01),
            BottleneckNoSE(64, 64, dropout_prob=0.01)
        )

        self.stage2 = nn.Sequential(
            BottleneckNoSE(64, 128, bottleneck_type='down', dropout_prob=0.1),
            BottleneckNoSE(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoSE(128, 128, asymmetric=True, dropout_prob=0.1)
        )

        self.stage3 = nn.Sequential(
            BottleneckNoSE(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoSE(128, 128, asymmetric=True, dropout_prob=0.1),
            BottleneckNoSE(128, 128, dilation=4, dropout_prob=0.1)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            BottleneckNoSE(64, 64, dropout_prob=0.1)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            BottleneckNoSE(16, 16, dropout_prob=0.1)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final(x)
        x = F.interpolate(x, size=(720, 1280), mode='bilinear', align_corners=True)
        return x

# --- CustomENet without DepthwiseSeparableConv ---
class CustomENetNoDW(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CustomENetNoDW, self).__init__()

        self.initial = InitialBlock(in_channels, 16)

        self.stage1 = nn.Sequential(
            BottleneckNoDW(16, 64, bottleneck_type='down', dropout_prob=0.01),
            BottleneckNoDW(64, 64, dropout_prob=0.01)
        )

        self.stage2 = nn.Sequential(
            BottleneckNoDW(64, 128, bottleneck_type='down', dropout_prob=0.1),
            BottleneckNoDW(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoDW(128, 128, asymmetric=True, dropout_prob=0.1)
        )

        self.stage3 = nn.Sequential(
            BottleneckNoDW(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoDW(128, 128, asymmetric=True, dropout_prob=0.1),
            BottleneckNoDW(128, 128, dilation=4, dropout_prob=0.1)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            BottleneckNoDW(64, 64, dropout_prob=0.1)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            BottleneckNoDW(16, 16, dropout_prob=0.1)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final(x)
        x = F.interpolate(x, size=(720, 1280), mode='bilinear', align_corners=True)
        return x

# --- CustomENet without Asymmetric Conv ---
class CustomENetNoAsym(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CustomENetNoAsym, self).__init__()

        self.initial = InitialBlock(in_channels, 16)

        self.stage1 = nn.Sequential(
            BottleneckNoAsym(16, 64, bottleneck_type='down', dropout_prob=0.01),
            BottleneckNoAsym(64, 64, dropout_prob=0.01)
        )

        self.stage2 = nn.Sequential(
            BottleneckNoAsym(64, 128, bottleneck_type='down', dropout_prob=0.1),
            BottleneckNoAsym(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoAsym(128, 128, dropout_prob=0.1)
        )

        self.stage3 = nn.Sequential(
            BottleneckNoAsym(128, 128, dilation=2, dropout_prob=0.1),
            BottleneckNoAsym(128, 128, dropout_prob=0.1),
            BottleneckNoAsym(128, 128, dilation=4, dropout_prob=0.1)
        )

        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            BottleneckNoAsym(64, 64, dropout_prob=0.1)
        )

        self.stage5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            BottleneckNoAsym(16, 16, dropout_prob=0.1)
        )

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final(x)
        x = F.interpolate(x, size=(720, 1280), mode='bilinear', align_corners=True)
        return x