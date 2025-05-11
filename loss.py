import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.to(DEVICE)
        self.vgg.eval()  
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.perceptual_layers = [3,8,15]
    
    def forward(self, output, target):
        VGG_MEAN = [0.485, 0.456, 0.406]
        VGG_STD  = [0.229, 0.224, 0.225]
        out_norm = (output - torch.tensor(VGG_MEAN, device=output.device)[None,:,None,None]) / torch.tensor(VGG_STD, device=output.device)[None,:,None,None]
        tgt_norm = (target - torch.tensor(VGG_MEAN, device=target.device)[None,:,None,None]) / torch.tensor(VGG_STD, device=target.device)[None,:,None,None]
        feat_outs = []
        feat_tgts = []
        x_out = out_norm; x_tgt = tgt_norm
        for i, layer in enumerate(self.vgg):
            x_out = layer(x_out); x_tgt = layer(x_tgt)
            if i in self.perceptual_layers:
                feat_outs.append(x_out); feat_tgts.append(x_tgt)
        perc_loss = 0.0
        for f_out, f_tgt in zip(feat_outs, feat_tgts):
            perc_loss += F.mse_loss(f_out, f_tgt)  
        return perc_loss

class AnimeLoss(nn.Module):
    def __init__(self, a = 0.1):
        super(AnimeLoss, self).__init__()
        self.a = a
        self.device = DEVICE
        model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
        model.eval()
        model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False

        self.body = model[0]
        head_layers = list(model[1].children())[:5]  
        self.head = nn.Sequential(*head_layers)

        self.peceptual_loss = PerceptualLoss()

        self.normalize = transforms.Normalize(
            mean=[0.7137, 0.6628, 0.6519],
            std=[0.2970, 0.3017, 0.2979]
        ) 

    def forward(self, output, target):
        if self.a != 1:
            perceptual_loss = self.peceptual_loss(output, target)
        else:
            perceptual_loss = 0
        
        output = (output + 1) / 2
        target = (target + 1) / 2

        output = F.interpolate(output, size=(360, 360), mode='bilinear', align_corners=False)
        target = F.interpolate(target, size=(360, 360), mode='bilinear', align_corners=False)
        output = torch.stack([self.normalize(img) for img in output])
        target = torch.stack([self.normalize(img) for img in target])

        with torch.no_grad():
            out_f = self.head(self.body(output))
            tgt_f = self.head(self.body(target))

        anime_loss = F.mse_loss(out_f, tgt_f)
    
        return  self.a*anime_loss + (1-self.a)*perceptual_loss