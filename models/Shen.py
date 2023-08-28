'''
This network is a reproduction of Shen Lili's Network.
The original paper is as follows:

L. Shen, X. Chen, Z. Pan, K. Fan, F. Li, and J. Lei, “No-reference stereoscopic image quality assessment 
based on global and local content characteristics,” Neurocomputing, vol. 424, pp. 132–142, 2021.

Author: Huilin Zhang (Fanning)

'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Shen_LocalGlobal(nn.Module):
    
    def __init__(self):
        super(Shen_LocalGlobal, self).__init__()

        # The Primary sub-network for L&R
        self.primary_subnet_left = PrimarySubNetwork()
        self.primary_subnet_right = PrimarySubNetwork()

        # The Local sub-network for L&R
        self.local_subnet_left = LocalSubNetwork()
        self.local_subnet_right = LocalSubNetwork()

        # The Global sub-network
        self.global_subnet = GlobalSubNetwork()

        # The Regression Module
        self.regression_module = RegressionModule()


    def forward(self, left_image, right_image):

        left_feature = self.primary_subnet_left(left_image)
        right_feature = self.primary_subnet_right(right_image)

        left_flatten = self.local_subnet_left(left_feature)
        right_flatten = self.local_subnet_right(right_feature)

        global_flatten = self.global_subnet(left_feature, right_feature)

        q = self.regression_module(left_flatten, global_flatten, right_flatten)
        
        return q


class PrimarySubNetwork(nn.Module):
    
    def __init__(self):
        super(PrimarySubNetwork, self).__init__()

        self.primary_subnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, image):

        extracted = self.primary_subnet(image)
        
        return extracted


class LocalSubNetwork(nn.Module):
    
    def __init__(self):
        super(LocalSubNetwork, self).__init__()

        self.local_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acb_1 = ACB(in_channels=64, out_channels=128)
        self.local_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acb_2 = ACB(in_channels=128, out_channels=256)
        self.local_pool_3 = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, single):

        single = self.local_pool_1(single)
        single = self.acb_1(single)
        single = self.local_pool_2(single)
        single = self.acb_2(single)
        single = self.local_pool_3(single)
        single_flatten = single.view(single.size(0), -1)
        
        return single_flatten


class GlobalSubNetwork(nn.Module):
    
    def __init__(self):
        super(GlobalSubNetwork, self).__init__()

        self.msp = MSP()
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, l_feature, r_feature):

        l_feature = torch.split(l_feature, [32,32], dim=1)
        r_feature = torch.split(r_feature, [32,32], dim=1)

        cross_fusion = torch.cat((l_feature[0], r_feature[0], l_feature[1], r_feature[1]), 1)

        gbl = self.msp(cross_fusion)
        gbl = self.conv7(gbl)
        gbl_flatten = gbl.view(gbl.size(0), -1)
        
        return gbl_flatten


class RegressionModule(nn.Module):
    
    def __init__(self):
        super(RegressionModule, self).__init__()

        self.weight_branch = nn.Sequential(    
            nn.Linear(1536, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.quality_branch = nn.Sequential(    
            nn.Linear(1536, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, flatten_left, flatten_global, flatten_right):

        flatten = torch.cat((flatten_left, flatten_global, flatten_right), 1)
        weight = self.weight_branch(flatten)
        quality = self.quality_branch(flatten)
        
        return weight*quality


class ACB(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ACB, self).__init__()

        # three types of convs for ACB block
        self.conv_1_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_3_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0))

    def forward(self, local_feature):

        # torch.add is equal to '+' manipulation.
        ACB_feature = self.conv_1_3(local_feature) + self.conv_3_3(local_feature) + self.conv_3_1(local_feature)
        
        return ACB_feature


class MSP(nn.Module):
    
    def __init__(self):
        super(MSP, self).__init__()

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, global_feature):

        feature_low = self.conv5(global_feature)
        feature_high = self.conv6(global_feature)
        msp_fusion = torch.cat((feature_low, feature_high), 1)
        
        return msp_fusion
    

# ----------------------------------------------------------------------------

# if __name__ == "__main__":

#     in_left_tensor = torch.ones(64, 3, 40, 40)
#     in_right_tensor = torch.ones(64, 3, 40, 40)

#     net = Shen_LocalGlobal()
#     out_tensor = net(in_left_tensor, in_right_tensor)

#     print(in_left_tensor.shape)
#     print(out_tensor.shape)
