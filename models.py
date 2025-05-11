import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1, activation='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            return F.relu(x, inplace=True)
        elif self.activation == 'leaky':
            return F.leaky_relu(x, negative_slope=0.2, inplace=True)
        else:
            return x

class ClassifierFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50', pretrained=False)
            model.load_state_dict(torch.load("../resnet50-13306192.pth", map_location='cpu'))
        else:
            model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
        model.eval()
        model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False

        self.body = model[0]
        head_layers = list(model[1].children())[:5]  #
        self.head = nn.Sequential(*head_layers)

        self.normalize = transforms.Normalize(
            mean=[0.7137, 0.6628, 0.6519],
            std=[0.2970, 0.3017, 0.2979]
        ) 

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x + 1) / 2
        
        x = F.interpolate(x, size=(360, 360), mode='bilinear', align_corners=False)
        x = torch.stack([self.normalize(img) for img in x])
        
        with torch.no_grad():
            x = self.body(x)
            x = self.head(x)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.ch_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W

        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # B x N x C'
        proj_key   = self.key_conv(x).view(B, -1, N)                     # B x C' x N
        energy = torch.bmm(proj_query, proj_key)                        # B x N x N
        attention = self.softmax(energy)                                # B x N x N

        proj_value = self.value_conv(x).view(B, -1, N)                  # B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # B x C x N
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out
    
class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x

class ColorizationUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, use_attention=False, use_extractor=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_attention = use_attention
        self.use_extractor = use_extractor
        self.down1 = ConvBlock(in_channels, 64, activation='leaky')   # 256 -> 128
        self.down2 = ConvBlock(64, 128, activation='leaky')           # 128 -> 64
        self.down3 = ConvBlock(128, 256, activation='leaky')          # 64 -> 32
        self.down4 = ConvBlock(256, 512, activation='leaky')          # 32 -> 16
        self.down5 = ConvBlock(512, 512, activation='leaky')          # 16 -> 8
        self.down6 = ConvBlock(512, 512, activation='leaky')          # 8 -> 4
        self.down7 = ConvBlock(512, 512, activation='leaky')          # 4 -> 2
        self.down8 = ConvBlock(512, 512, activation='leaky')          # 2 -> 1 (bottleneck)
        if self.use_extractor:
            self.context = ClassifierFeatureExtractor()  
        if self.use_attention:
            self.attn = SelfAttention(512*2 if self.use_extractor else 512) 

        self.up1 = DeconvBlock(512*2 if self.use_extractor else 512, 512, dropout=True)  # 1 -> 2
        self.up2 = DeconvBlock(512*2, 512, dropout=True) # 2 -> 4 
        self.up3 = DeconvBlock(512*2, 512, dropout=True) # 4 -> 8 
        self.up4 = DeconvBlock(512*2, 512)  # 8 -> 16 
        self.up5 = DeconvBlock(512*2, 256)  # 16 -> 32 
        self.up6 = DeconvBlock(256*2, 128)  # 32 -> 64 
        self.up7 = DeconvBlock(128*2, 64)   # 64 -> 128 
        self.up8 = nn.ConvTranspose2d(64*2, out_channels, kernel_size=4, stride=2, padding=1)  # 128->256


    def forward(self, x):
        d1 = self.down1(x)   # -> (64, 128,128)
        d2 = self.down2(d1)  # -> (128, 64,64)
        d3 = self.down3(d2)  # -> (256, 32,32)
        d4 = self.down4(d3)  # -> (512, 16,16)
        d5 = self.down5(d4)  # -> (512, 8,8)
        d6 = self.down6(d5)  # -> (512, 4,4)
        d7 = self.down7(d6)  # -> (512, 2,2)
        d8 = self.down8(d7)  # -> (512, 1,1)
        bottleneck = d8
        if self.use_extractor:
            c = self.context(x)    
            c = c.unsqueeze(2).unsqueeze(3)     
            bottleneck = torch.cat([bottleneck, c], dim=1)  

        if self.use_attention:
            bottleneck = self.attn(bottleneck) 
        u1 = self.up1(bottleneck)               # -> (512, 2,2)
        u1 = torch.cat([u1, d7], dim=1)         # concat skip from down7 (512 -> 1024 ch)
        u2 = self.up2(u1)                       # -> (512, 4,4)
        u2 = torch.cat([u2, d6], dim=1)         # concat skip from down6
        u3 = self.up3(u2)                       # -> (512, 8,8)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)                       # -> (512, 16,16)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)                       # -> (256, 32,32)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)                       # -> (128, 64,64)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)                       # -> (64, 128,128)
        u7 = torch.cat([u7, d1], dim=1)
        out = self.up8(u7)                      # -> (3, 256,256)

        out = torch.tanh(out)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1+3): 
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)   # 256->128
        self.conv2 = ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, activation='leaky')  # 128->64
        self.conv3 = ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, activation='leaky') # 64->32
        self.conv4 = ConvBlock(256, 512, kernel_size=4, stride=2, padding=1, activation='leaky') # 32->16
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 16->15 (roughly)
    def forward(self, sketch, color_image):
        x = torch.cat([sketch, color_image], dim=1)  # shpe: (batch, 1+3, 256,256)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.conv2(x)  
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
