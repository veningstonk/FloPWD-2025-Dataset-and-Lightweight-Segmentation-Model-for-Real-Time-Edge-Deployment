import torch
import torch.nn as nn

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

class ENet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dropout_probs=None, 
                 dilations_stage2=None, dilations_stage3=None, use_asymmetric=True, 
                 blocks=None, channel_factor=4):
        super(ENet, self).__init__()
        
        # Default configurations
        dropout_probs = [0.01, 0.1, 0.1, 0.1, 0.1] if dropout_probs is None else dropout_probs
        dilations_stage2 = [2, 4, 8] if dilations_stage2 is None else dilations_stage2
        dilations_stage3 = [2, 4, 8, 16] if dilations_stage3 is None else dilations_stage3
        blocks = [5, 8, 8, 2, 1] if blocks is None else blocks
        
        self.initial = InitialBlock(in_channels, 16)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, bottleneck_type='down', dropout_prob=dropout_probs[0],
                      internal_channels=64 // channel_factor),
            *[Bottleneck(64, 64, dropout_prob=dropout_probs[0],
                        internal_channels=64 // channel_factor) for _ in range(blocks[0] - 1)]
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, bottleneck_type='down', dropout_prob=dropout_probs[1],
                      internal_channels=128 // channel_factor),
            Bottleneck(128, 128, dropout_prob=dropout_probs[1],
                      internal_channels=128 // channel_factor),
            *[Bottleneck(128, 128, dilation=dilations_stage2[i % len(dilations_stage2)], 
                        asymmetric=use_asymmetric if i in [2, 4] else False, 
                        dropout_prob=dropout_probs[1],
                        internal_channels=128 // channel_factor) for i in range(1, blocks[1])]
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128, dropout_prob=dropout_probs[2],
                      internal_channels=128 // channel_factor),
            *[Bottleneck(128, 128, dilation=dilations_stage3[i % len(dilations_stage3)], 
                        asymmetric=use_asymmetric if i in [1, 3, 5] else False, 
                        dropout_prob=dropout_probs[2],
                        internal_channels=128 // channel_factor) for i in range(1, blocks[2])]
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            *[Bottleneck(64, 64, dropout_prob=dropout_probs[3],
                        internal_channels=64 // channel_factor) for _ in range(blocks[3])]
        )
        
        # Stage 5
        self.stage5 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            *[Bottleneck(16, 16, dropout_prob=dropout_probs[4],
                        internal_channels=16 // channel_factor) for _ in range(blocks[4])]
        )
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(1280, 720), mode='bilinear', align_corners=True)
        
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