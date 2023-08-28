'''
This network is a reproduction of Fang's Network: Siamese network.
The original paper is as follows:

Y. Fang, J. Yan, X. Liu, and J. Wang, “Stereoscopic image quality assessment by deep convolutional neural network,” 
Journal of Visual Communication and Image Representation, vol. 58, pp. 400–406, 2019.

Author: Huilin Zhang (Fanning)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fang_SiameseNetwork(nn.Module):
    
    def __init__(self):
        super(Fang_SiameseNetwork, self).__init__()

        # The first convolution block for L&R
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The second convolution block for L&R
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )
        
        # The third convolution block for L&R
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)            
        )

        # The fourth convolution block for L&R
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # The convolution block after concat
        self.conv_after_concat = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC block
        self.fc = nn.Sequential(    
            nn.Linear(2*2*512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, left_image, right_image):

        # 4 groups for L&R image branch
        left_image = self.conv1(left_image)
        right_image = self.conv1(right_image)
        
        left_image = self.conv2(left_image)
        right_image = self.conv2(right_image)   

        left_image = self.conv3(left_image)
        right_image = self.conv3(right_image)

        left_image = self.conv4(left_image)
        right_image = self.conv4(right_image)

        # Concat
        fusion = torch.cat((left_image, right_image), 1)
        
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

#     net = Fang_SiameseNetwork()
#     out_tensor = net(in_left_tensor, in_right_tensor)

#     print(in_left_tensor.shape)
#     print(out_tensor.shape)
