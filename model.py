import torch
import torch.nn as nn

# import utils
from utils.modelutils import *

class LightFlow(nn.Module):
    def __init__(self, in_channels=6):
        super(LightFlow, self).__init__()

        # Encoder Network
        
        self.conv1 = DWConv(in_channels, 32, 2)
        self.conv2 = DWConv(32, 64, 2)
        self.conv3 = DWConv(64, 128, 2)
        self.conv4a = DWConv(128, 256, 2)
        self.conv4b = DWConv(256, 256, 1)
        self.conv5a = DWConv(256, 512, 2)
        self.conv5b = DWConv(512, 512, 1)
        self.conv6a = DWConv(512, 1024, 2)
        self.conv6b = DWConv(1024, 1024, 1)

        # Decoder Network

        self.conv7 = DWConv(1024, 256, 1)
        self.conv8 = DWConv(768, 128, 1)
        self.conv9 = DWConv(384, 64, 1)
        self.conv10 = DWConv(192, 32, 1)
        self.conv11 = DWConv(96, 16, 1)

        # Optical Flow Predictions

        self.conv12 = DWConv(256, 2, 1)
        self.conv13 = DWConv(128, 2, 1)
        self.conv14 = DWConv(64, 2, 1)
        self.conv15 = DWConv(32, 2, 1)
        self.conv16 = DWConv(16, 2, 1)

        # Up Sampling operation
        self.up = UP()

        # Average layer
        self.average = Average()

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1, x2), dim = 1)

        ##### Encoder #####
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4a = self.conv4a(conv3)
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)
        conv5b = self.conv5b(conv5a)
        conv6a = self.con6a(conv5b)
        conv6b = self.conv6b(conv6a)

        ##### Decoder #####
        conv7 = self.conv7(conv6b)
        
        # 1st Up Sampling and Concat
        conv7_x2 = self.up(conv7, 2)
        concat_1 = torch.cat([conv7_x2, conv5b], dim=1)

        conv8 = self.conv8(concat_1)
        
        # 2nd Up Sampling and Concat
        conv8_x2 = self.up(conv8, 2)
        concat_2 = torch.cat([conv8_x2, conv4b], dim=1)

        conv9 = self.conv9(concat_2)

        # 3rth Up Sampling and Concat
        conv9_x2 = self.up(conv9, 2)
        concat_3 = torch.cat([conv9_x2, conv3], dim=1)

        conv10 = self.conv10(concat_3)

        # 4rth Up Sampling and Concat
        conv10_x2 = self.up(conv10, 2)
        concat_4 = torch.cat([conv10_x2, conv2], dim=1)

        conv11 = self.conv11(concat_4)

        ##### Optical Flow predictions #####

        conv12 = self.conv12(conv7)
        conv13 = self.conv13(conv8)
        conv14 = self.conv14(conv9)
        conv15 = self.conv15(conv10)
        conv16 = self.conv16(conv11)

        conv12_x16 = self.up(conv12, 16)
        conv13_x8  = self.up(conv13, 8)
        conv14_x4  = self.up(conv14, 4)
        conv15_x2 = self.up(conv15, 2)

        average = self.average([conv12_x16, conv13_x8, conv14_x4, conv15_x2, conv16])
        return average

if __name__ == "__main__":
    model = LightFlow()
    print(model)