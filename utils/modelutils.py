import torch.nn as nn
import torch

class DWConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(DWConv, self).__init__()

        self.dwconv =  nn.Sequential(
            # DepthWise Convolution
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            
            # PointWise Convolution
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x

class UP(nn.Module):
    def __init__(self, scale=2, bilinear=True):
        super(UP, self).__init__()
        self.scale = scale
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=True)

    def forward(self, x, scale):
        self.scale = scale
        x = self.up(x)
        return x

class Average(nn.Module):
    """
    Layer that averages a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def __init__(self):
        super(Average, self).__init__()
    def forward(self, x):
        # return torch.sum(x)/len(x)
        output = x[0]
        for i in range(1, len(x)):
            output += x[i]
        return output / len(x)



"""
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
"""
