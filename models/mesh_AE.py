import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.utils import ico_sphere

class PointEncoder(nn.Module):

    def __init__(self, out_ch=32):
        super(PointEncoder, self).__init__()
        
        channels = [512, 512, 512, 512]
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, channels[0], kernel_size=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=1),
            nn.ReLU())
        
        self.out_layer = nn.Conv1d(channels[2], out_ch, kernel_size=1)
    
    def forward(self, x):
        '''
        Input:
            x: [B,N,3]
        Output:
            x: [B,C], latent code
        '''
        x = x.permute(0,2,1)

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.max(x, dim=2, keepdim=True)[0]

        x = self.out_layer(x)
        x = x.squeeze(-1)

        x = F.normalize(x, dim=1)
        return x
    
class PointDecoder(nn.Module):

    def __init__(self, input_ch=32, out_ch=3*42, voxel_size=0.5):
        super(PointDecoder, self).__init__()

        self.voxel_size = voxel_size

        channels = [512, 512, 512, 512]
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_ch, channels[0], kernel_size=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=1),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(channels[2], channels[3], kernel_size=1),
            nn.ReLU())
        
        self.out_layer = nn.Conv1d(channels[3], out_ch, kernel_size=1)
    
    def forward(self, x):
        '''
        Input:
            x: [B,C], latent code
        Output:
            x: [B,42,3], vertex coordinates
        '''
        bs = x.shape[0]
        sphere_mesh = ico_sphere(level=1, device=x.device)
        template_verts = sphere_mesh.verts_packed() * self.voxel_size
        template_verts = template_verts.unsqueeze(0).repeat(bs, 1, 1)
        x = x.unsqueeze(-1)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.out_layer(x)

        x = x.view(bs, -1, 3).contiguous()
        x = x + template_verts

        return x