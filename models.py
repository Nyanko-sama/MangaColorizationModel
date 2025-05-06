import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from datasets import TestVis
import os
import csv
import torchvision.models as models
import torchvision.transforms as transforms
from loss import GANLoss

class Downsample(torch.nn.Module):
    def __init__(self, filters, size, apply_batchnorm=True):
        super().__init__()
        self.downsample = torch.nn.Sequential()
        self.downsample.add_module('Conv2D', torch.nn.Conv2d(filters, size, kernel_size=4, stride=2, padding=1, bias=False))
        
        if apply_batchnorm:
            self.downsample.add_module('batchnorm', torch.nn.BatchNorm2d(size))
        self.downsample.add_module('leakyrelu', torch.nn.LeakyReLU())

    
    def forward(self, x):
        return self.downsample(x)

class Upsample(torch.nn.Module):
    def __init__(self, filters, size, apply_dropout=False):
        super().__init__()
        self.upsample = torch.nn.Sequential()
        self.upsample.add_module('Conv2DTranspose', torch.nn.ConvTranspose2d(filters, size, kernel_size=4, stride=2, padding=1, bias=False))
        self.upsample.add_module('batchnorm', torch.nn.BatchNorm2d(size))

        if apply_dropout:
            self.upsample.add_module('dropout', torch.nn.Dropout(0.5))
        
        self.upsample.add_module('relu', torch.nn.ReLU())

    def forward(self, x):
        return self.upsample(x)

class SmallerUnet(torch.nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        self.out_channels = out_channels

        # Downsampling block (in_channels, out_channels)
        self.down1 = Downsample(1, 32, apply_batchnorm=False) # 256 -> 128
        self.down2 = Downsample(32, 64) # 128 -> 64
        self.down3 = Downsample(64, 128)  # 64 -> 32
        self.down4 = Downsample(128, 256) # 32 -> 16
        self.down5 = Downsample(256, 512) # 16 -> 8
        self.down6 = Downsample(512, 512) # 8 -> 4
        self.down7 = Downsample(512, 512) # 4 -> 2

        # Upsampling block (in_channels, out_channels)
        self.up1 = Upsample(512, 512, apply_dropout=True) # 2 -> 4
        self.up2 = Upsample(1024, 512, apply_dropout=True) # 4 -> 8
        self.up3 = Upsample(1024, 512, apply_dropout=True) # 8 -> 16
        self.up4 = Upsample(768, 256) # 16 -> 32
        self.up5 = Upsample(384, 256) # 32 -> 64
        self.up6 = Upsample(256+64, 128) # 64 -> 128
        self.up7 = Upsample(128+32, 64) # 128 -> 256

        self.final = torch.nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Upsampling with skip connections
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        u7 = self.up7(torch.cat([u6, d1], dim=1))

        out = self.final(u7)
        return self.tanh(out)   

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor = 1, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*factor, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels*factor),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels*factor, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x_1 = self.block(x)
        x = x + x_1
        x = self.relu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding = 1)
        self.res = ResBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
        )
        self.conv = ResBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        return x

class ResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.down1 = DownBlock(in_channels, 64)  # 256 -> 128
        self.down2 = DownBlock(64, 128) # 128 -> 64
        self.down3 = DownBlock(128, 256)  # 64 -> 32
        self.down4 = DownBlock(256, 512)       # 32 -> 16

        self.down5 = ResBlock(512, 512) 
        self.up1 = ResBlock(512, 512)   

        self.up2 = UpBlock(1024, 256)  # 16 -> 32
        self.up3 = UpBlock(512, 128) # 32 -> 64
        self.up4 = UpBlock(256, 64)  # 64 -> 128
        self.up5 = UpBlock(128, 64) # 128 -> 256

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x_1 = self.down1(x)
        x_2 = self.down2(x_1)
        x_3 = self.down3(x_2)
        x_4 = self.down4(x_3)
        x_5 = self.down5(x_4)

        x_6 = self.up1(x_5)
        x_6 = torch.cat((x_6, x_4), dim=1)
        x_7 = self.up2(x_6)
        x_7 = torch.cat((x_7, x_3), dim=1)
        x_8 = self.up3(x_7)
        x_8 = torch.cat((x_8, x_2), dim=1)
        x_9 = self.up4(x_8)
        x_9 = torch.cat((x_9, x_1), dim=1)
        x_10 = self.up5(x_9)

        return self.final_conv(x_10)


class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.efficientnet_b0(pretrained=True)
        self.features = resnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling

        self.global_conv = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
        )
        # freeze the parameters of the EfficientNet model
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) # Convert grayscale to RGB by repeating the channel 3 times
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False) # Resize to 224x224 for EfficientNet

        # ensure that even batchnorm layers are in eval mode
        # and that the model is not training
        self.features.eval()
        with torch.no_grad():
            x = self.features(x)  # [B, 1280, 7, 7]

        x = self.pool(x)      # [B, 1280, 1, 1]
        x = self.global_conv(x)  # [B, 512, 16, 16]

        return x
    
class FusionNet(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        self.mid_feat = nn.ModuleList([
            DownBlock(1, 64), # 256 -> 128
            DownBlock(64, 128), # 128 -> 64
            DownBlock(128, 256), # 64 -> 32
            DownBlock(256, 512), # 32 -> 16
            ResBlock(512, 512), # 16 -> 16
        ])

        self.classification = Classification()

        self.colorization = nn.ModuleList([
            ResBlock(1024, 1024), # 16 -> 16
            UpBlock(1536, 512), # 16 -> 32
            UpBlock(768, 256), # 32 -> 64
            UpBlock(256+128, 128), # 64 -> 128
            UpBlock(128+64, 64), # 128 -> 256
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        d_1 = self.mid_feat[0](x)
        d_2 = self.mid_feat[1](d_1)
        d_3 = self.mid_feat[2](d_2)
        d_4 = self.mid_feat[3](d_3)
        mid_feat = self.mid_feat[4](d_4)    # [B, 512, 16, 16]

        global_feat = self.classification(x)  # [B, 512, 16, 16]

        fused = torch.cat([mid_feat, global_feat], dim=1)  # [B, 1024, 16, 16]

        u_1 = self.colorization[0](fused)  # [B, 512, 16, 16]
        u_2 = self.colorization[1](torch.cat((u_1, d_4), dim=1))    # [B, 256, 32, 32]
        u_3 = self.colorization[2](torch.cat((u_2, d_3), dim=1))   # [B, 128, 64, 64]
        u_4 = self.colorization[3](torch.cat((u_3, d_2), dim=1))  # [B, 64, 128, 128]
        u_5 = self.colorization[4](torch.cat((u_4, d_1), dim=1)) # [B, 64, 256, 256]
        x = self.final_conv(u_5)

        return x



def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

# from https://github.com/mberkay0/image-colorization
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  # the 'if' statement is taking care of not using
        # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  # Make sure to not use normalization or
        # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  # when needing to make some repeatitive blocks of layers,
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)]  # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# from https://github.com/mberkay0/image-colorization
class GANModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data[0].to(self.device)
        self.ab = data[1].to(self.device)

    def forward(self):
        self.net_G = self.net_G.to(self.device)
        self.L = self.L.to(self.device)
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

# from https://github.com/richzhang/colorization/blob/master/colorizers/eccv16.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1,2), dilation=1):
        super().__init__()
        self.layer = nn.Sequential()
        pad = 1 if dilation == 1 else 2
        for i, stride in enumerate(strides):
            self.layer.add_module(f'conv_{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=pad, bias=True, dilation=dilation))
            self.layer.add_module(f'relu_{i}', nn.ReLU(True))
            in_channels = out_channels
        self.layer.add_module('batchnorm', nn.BatchNorm2d(out_channels))
    
    def forward(self, x):
        return self.layer(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = ConvBlock(1, 64)
        self.l2 = ConvBlock(64, 128)
        self.l3 = ConvBlock(128, 256, strides=(1,1,2))
        self.l4 = ConvBlock(256, 512, strides=(1,1,1))
        self.l5 = ConvBlock(512, 512, strides=(1,1,1), dilation=2)
        self.l6 = ConvBlock(512, 512, strides=(1,1,1), dilation=2)
        self.l7 = ConvBlock(512, 512, strides=(1,1,1))

        self.l8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        conv1_2 = self.l1(x)
        conv2_2 = self.l2(conv1_2)
        conv3_3 = self.l3(conv2_2)
        conv4_3 = self.l4(conv3_3)
        conv5_3 = self.l5(conv4_3)
        conv6_3 = self.l6(conv5_3)
        conv7_3 = self.l7(conv6_3)
        conv8_3 = self.l8(conv7_3)
        out_reg = self.out(self.softmax(conv8_3))
        out_reg = self.upsample4(out_reg)
        return out_reg


