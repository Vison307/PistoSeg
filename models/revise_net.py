import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.resnet38d

class Net(models.resnet38d.Net):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        
        self.f9_1 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        self.f9_2 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9_1.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.f9_2.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9_1, self.f9_2, self.fc8]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        
    def get_norm_cam_d(self, cam):
        """normalize the activation vectors of each pixel by supressing foreground non-maximum activations to zeros"""
        n, c, h, w = cam.size() # [2, 4, 32, 32]
        with torch.no_grad():
            cam_d = cam.detach()
            cam_d_min = torch.min(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1)
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5 # [2, 4, 1, 1] each channel has a max value (for each image in batch)
            cam_d_norm = (cam - cam_d_min) / (cam_d_max - cam_d_min) # [2, 4, 32, 32]
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0] # background channel is 0, which is calculated by 1 - max(other channels)
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0] # [2, 1, 32, 32], max value of each channel (except background channel)
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0 # set the non-max value to 0
            
        return cam_d_norm

    def forward(self, x, pmask, pcam):
        N, C, H, W = x.size()  # [B, 3, 256, 256]
        d = super().forward_as_dict(x)
        assert not d['conv4'].isnan().any()
        assert not d['conv5'].isnan().any()
        assert not d['conv6'].isnan().any()

        cam = self.fc8(self.dropout7(d['conv6'])) # d['conv6']: [2, 4096, 32, 32], cam: [2, 4, 32, 32]
        n, c, h, w = cam.size() # [2, 4, 28, 28]

        cam_d_norm = self.get_norm_cam_d(cam)
        pmask_d_norm = self.get_norm_cam_d(pmask)
        pcam_d_norm = self.get_norm_cam_d(pcam)

        # ----> Get Concated Feature
        # f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True) # d['conv4']: [2, 512, 32, 32], f8_3: [2, 64, 32, 32]
        # f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True) # d['conv5']: [2, 1024, 32, 32], f8_4: [2, 128, 32, 32]
        
        f8_3 = F.relu(self.f8_3(d['conv4']), inplace=True) # d['conv4']: [2, 512, 32, 32], f8_3: [2, 64, 32, 32]
        f8_4 = F.relu(self.f8_4(d['conv5']), inplace=True) # d['conv5']: [2, 1024, 32, 32], f8_4: [2, 128, 32, 32]
        
        x_s = F.interpolate(x, (h, w), mode='bilinear',align_corners=True) # x_s: [2, 3, 32, 32]
        f = torch.cat([x_s, f8_3, f8_4], dim=1) # [2, 192+3, 32, 32]
        n, c, h, w = f.size() # [2, 192+3, 32, 32]
        
        # ----> Attention
        q = self.f9_1(f).view(n, -1, h*w) # [2, 192, 32*32]
        # q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-5)
        k = self.f9_2(f).view(n, -1, h*w) # [2, 192, 32*32]
        # k = k / (torch.norm(k, dim=1, keepdim=True) + 1e-5)
        A = torch.matmul(q.transpose(1, 2), k) # [2, 32*32, 32*32]
        A = F.softmax(A, dim=1) # normalize over column
        assert not torch.isnan(A).any(), A
        
        pmask_refine = self.RFM(pmask_d_norm, A, h, w)
        pmask_rv = F.interpolate(pmask_refine, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]
        
        pcam_refine = self.RFM(pcam_d_norm, A, h, w)
        pcam_rv = F.interpolate(pcam_refine, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]
        
        cam_refine = self.RFM(cam_d_norm, A, h, w)
        cam_rv = F.interpolate(cam_refine, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]
        
        cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True) # [2, 4, 256, 256]
        
        return cam, cam_rv, pmask_rv, pcam_rv

    def RFM(self, cam, A, h=32, w=32): 
        n = A.size()[0]
        
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h*w) # [2, 4, h, w] -> [2, 4, 32*32]
        cam_rv = torch.matmul(cam, A).view(n, -1, h, w)
        
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

