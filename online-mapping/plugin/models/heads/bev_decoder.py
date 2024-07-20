import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from mmdet.models import HEADS
import torch.nn.functional as F

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


@HEADS.register_module(force=True)
class BevDecoder(nn.Module):
    def __init__(self, inC=256, hC=256, outC=3):
        super().__init__()
        
        self.up1 = Up(inC, hC, scale_factor=2)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hC, hC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hC),
            nn.ReLU(inplace=True),
            nn.Conv2d(hC, outC, kernel_size=3, padding=1)
        )
    
    def init_weights(self):
        for m in self.up1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.up2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.up1(x)
        x = F.relu(x, inplace=True)
        x = self.up2(x)

        return x

    def post_process(self, preds, tokens, thr=0.0):
        bs = len(preds)

        results = []
        for i in range(bs):
            single_result = {
                'predict_mask': (preds[i] > thr).detach().cpu().numpy(),
                'token': tokens[i]
            }
            results.append(single_result)
        
        return results