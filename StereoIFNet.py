'''
This network is a reproduction of Si's Network: StereoIFNet.
The original paper is as follows:

J. Si, B. Huang, H. Yang, W. Lin and Z. Pan, "A no-Reference Stereoscopic Image Quality Assessment Network Based on 
Binocular Interaction and Fusion Mechanisms," in IEEE Transactions on Image Processing, vol. 31, pp. 3066-3080, 2022.

Author: Huilin Zhang (Fanning)

'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoIFNet(nn.Module):
    
    def __init__(self):
        super(StereoIFNet, self).__init__()

        # The Primary network for L&R
        self.primary_left = PrimaryNetwork()
        self.primary_right = PrimaryNetwork()

        # BIM, the output channels of bim1, 2, 3, 4 is 768, 768, 1536, 1536, respectively.
        self.bim1 = BIM(in_ch=64, Nac_conv4_ch=384, Ac_ch=192, conv3_ch=32)
        self.bim2 = BIM(in_ch=768, Nac_conv4_ch=384, Ac_ch=192, conv3_ch=32)
        self.bim3 = BIM(in_ch=768, Nac_conv4_ch=768, Ac_ch=384, conv3_ch=64)
        self.bim4 = BIM(in_ch=1536, Nac_conv4_ch=768, Ac_ch=384, conv3_ch=64)

        # BFM
        self.bfm = BFM()

        # FC block
        self.fc = nn.Sequential(    
            nn.Linear(40*40*1536, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, left_image, right_image):

        # The Primary network for L&R
        left_feature = self.primary_left(left_image)
        right_feature = self.primary_right(right_image)

        # BIM
        left_feature, right_feature = self.bim1(left_feature, right_feature)
        left_feature, right_feature = self.bim2(left_feature, right_feature)
        left_feature, right_feature = self.bim3(left_feature, right_feature)
        left_feature, right_feature = self.bim4(left_feature, right_feature)

        # BFM
        fusion = self.bfm(left_feature, right_feature)

        # FC block
        q = self.fc(fusion)
        
        return q


class PrimaryNetwork(nn.Module):
    
    def __init__(self):
        super(PrimaryNetwork, self).__init__()

        self.primary = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, image):

        extracted = self.primary(image)
        
        return extracted


class BIM(nn.Module):
    
    def __init__(self, in_ch, Nac_conv4_ch, Ac_ch, conv3_ch):
        super(BIM, self).__init__()

        self.ConvNac_l = nn.Sequential(
            nn.Conv2d(in_ch, Nac_conv4_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Nac_conv4_ch)
        )
        
        self.ConvNac_r = nn.Sequential(
            nn.Conv2d(in_ch, Nac_conv4_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Nac_conv4_ch)
        )

        self.ConvAc_l = nn.Sequential(
            nn.Conv2d(in_ch, Ac_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Ac_ch),
            nn.ReLU()
        )

        self.ConvAc_r = nn.Sequential(
            nn.Conv2d(in_ch, Ac_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Ac_ch),
            nn.ReLU()
        )

        # 12 * conv3, 6 for summation, 6 for diffience.
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv3_ch, conv3_ch, kernel_size=1),
            nn.ReLU()
        )

        self.conv3s_sum = nn.ModuleList()
        self.conv3s_dif = nn.ModuleList()

        for _ in range(6):
            self.conv3s_sum.append(self.conv3)
            self.conv3s_dif.append(self.conv3)

        self.conv4 = nn.Sequential(
            nn.Conv2d(Nac_conv4_ch, Nac_conv4_ch, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, left, right):

        org_left = self.ConvNac_l(left)
        org_right = self.ConvNac_r(right)
        
        left = self.ConvAc_l(left)
        right = self.ConvAc_r(right)

        left = torch.split(left, int(left.size(1)/6), dim=1)
        right = torch.split(right, int(right.size(1)/6), dim=1)
        
        summation = []
        diffience = []

        # 6 conv3 for summation.
        for i, conv3 in enumerate(self.conv3s_sum):
            summation.append(conv3(left[i] + right[i]))

        # 6 conv3 for diffience.
        for i, conv3 in enumerate(self.conv3s_dif):
            diffience.append(conv3(left[i] - right[i]))
        
        sum_con = torch.cat((summation[0], summation[1], summation[2], summation[3], summation[4], summation[5]), dim=1)
        dif_con = torch.cat((diffience[0], diffience[1], diffience[2], diffience[3], diffience[4], diffience[5]), dim=1)
        I_con = torch.cat((sum_con, dif_con), dim=1)

        I_con = self.conv4(I_con)

        left_out = torch.cat((org_left, I_con), dim=1)
        right_out = torch.cat((I_con, org_right), dim=1)
        
        return left_out, right_out


class BFM(nn.Module):
    
    def __init__(self):
        super(BFM, self).__init__()

        self.weight_left_1 = nn.Sequential(    
            nn.Linear(1536*40*40, 512),
            nn.Dropout(0.5)
        )

        self.weight_left_2 = nn.Linear(512, 1)

        self.weight_right_1 = nn.Sequential(    
            nn.Linear(1536*40*40, 512),
            nn.Dropout(0.5)
        )

        self.weight_right_2 = nn.Linear(512, 1)

    def forward(self, left, right):

        left = left.view(left.size(0), -1)
        right = right.view(right.size(0), -1)

        weight_l = self.weight_left_1(left)
        weight_r = self.weight_right_1(right)

        weight_l = weight_l.view(weight_l.size(0), -1)
        weight_r = weight_r.view(weight_r.size(0), -1)

        weight_l = self.weight_left_2(weight_l)
        weight_r = self.weight_right_2(weight_r)
        
        return left * weight_l + right * weight_r


# ----------------------------------------------------------------------------

# if __name__ == "__main__":

#     in_left_tensor = torch.ones(64, 3, 40, 40)
#     in_right_tensor = torch.ones(64, 3, 40, 40)

#     net = StereoIFNet()
#     out_tensor = net(in_left_tensor, in_right_tensor)

#     print(in_left_tensor.shape)
#     print(out_tensor.shape)
