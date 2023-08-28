'''
This network is a reproduction of Yan Jiebin's network: MLFF.
The original paper is as follows:

J. Yan, Y. Fang, L. Huang, X. Min, Y. Yao, and G. Zhai, “Blind stereoscopic image quality assessment by deep neural network of multi-level feature fusion,” 
in 2020 IEEE International Conference on Multimedia and Expo (ICME), 2020, pp. 1–6.

Author: Huilin Zhang (Fanning)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Yan_MLFF(nn.Module):
    
    def __init__(self):
        super(Yan_MLFF, self).__init__()

        # The first convolution block for L&R feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The second convolution block for L&R feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        
        # The third convolution block for L&R feature extraction
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )

        # The convolution block for low-level branch
        self.conv_low = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # The convolution block for mid-level branch
        self.conv_mid = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # The convolution block for high-level branch
        self.conv_high = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # The C+P for fusion feature after concat
        self.conv_after_concat = nn.Sequential(
            nn.Conv2d(512*3, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC block
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, left_img, right_img):

        # 3 groups for L&R img branch
        left_img_low = self.conv1(left_img)
        right_img_low = self.conv1(right_img)
        
        left_img_mid = self.conv2(left_img_low)
        right_img_mid = self.conv2(right_img_low)   

        left_img_high = self.conv3(left_img_mid)
        right_img_high = self.conv3(right_img_mid)
        
        # low-level branch
        fusion_low = torch.cat((left_img_low, right_img_low), 1)
        fusion_low = self.conv_low(fusion_low)

        # mid-level branch
        fusion_mid = torch.cat((left_img_mid, right_img_mid), 1)
        fusion_mid = self.conv_mid(fusion_mid)

        # high-level branch
        fusion_high = torch.cat((left_img_high, right_img_high), 1)
        fusion_high = self.conv_high(fusion_high)

        # concat of three levels feature
        fusion = torch.cat((fusion_high, fusion_mid, fusion_low), 1)

        # C+P after concat
        fusion = self.conv_after_concat(fusion)

        # FC layer
        q = fusion.view(fusion.size(0), -1)
        q = self.fc(q)
        
        return q


# ----------------------------------------------------------------------------

# if __name__ == "__main__":

#     in_left_tensor = torch.ones(64, 3, 40, 40)
#     in_right_tensor = torch.ones(64, 3, 40, 40)

#     net = Yan_MLFF()
#     out_tensor = net(in_left_tensor, in_right_tensor)

#     print(in_left_tensor.shape)
#     print(out_tensor.shape)
