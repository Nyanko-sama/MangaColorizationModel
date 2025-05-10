import torch
from tqdm import tqdm
import os
import csv


import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn as nn

def save_val_collage(model, val_loader, device, save_path="collage.png"):
    model.eval()
    to_pil = T.ToPILImage()

    sketches, gt_colors = [], []

    val_data = []
    for sketch, color in val_loader:
        for i in range(sketch.size(0)):
            val_data.append((sketch[i], color[i]))
            if len(val_data) >= 10:
                break
        if len(val_data) >= 10:
            break
    
    pred = []
    with torch.no_grad():
        for idx in range(5):
            sketch, color = val_data[idx]
            sketch = sketch.unsqueeze(0).to(device)  
            color = color.unsqueeze(0).to(device)    

            sketch_disp = sketch[0].cpu().squeeze(0)
            color_disp = (color[0] + 1) / 2

            out = model(sketch)
            out = (out + 1) / 2

            # Save images
            sketches.append(to_pil(sketch_disp))
            gt_colors.append(to_pil(color_disp.cpu()))
            pred.append(to_pil(out[0].cpu()))

    # ---- Create 5×3 collage ----
    fig, axs = plt.subplots(5, 3, figsize=(15, 15))
    for i in range(5):
        axs[i, 0].imshow(sketches[i], cmap="gray")
        axs[i, 0].set_title("Sketch")
        axs[i, 1].imshow(pred[i])
        axs[i, 1].set_title("Predicted")
        axs[i, 2].imshow(gt_colors[i])
        axs[i, 2].set_title("Original Color")
        for j in range(3):
            axs[i, j].axis("off")


    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Collage saved to {save_path}")

class Trainer:
    def __init__(self, g_optimizer, d_optimizer, g_loss = None, device=None):
        self.device = device
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.perceptual_loss = g_loss
        self.adv_loss_fn = nn.BCEWithLogitsLoss()
        self.pixel_loss_fn = nn.L1Loss()


    def save_model(self, model, epoch, path):
        torch.save(model.state_dict(), os.path.join(path, f"model_epoch_{epoch}.pth"))

    def train(self, G, D, train_loader, validation_loader, num_epochs, path):
        losses_g = []
        losses_d = []
        for epoch in range(num_epochs):
            for batch in tqdm(train_loader):  
                sketch, color = batch
                sketch = sketch.to(self.device)    
                color = color.to(self.device)     

                # ------- Train Generator -------
                G.train(); D.train()
                fake_color = G(sketch)  
                fake_pred = D(sketch, fake_color)  
                real_label = torch.ones_like(fake_pred)
                g_adv_loss = self.adv_loss_fn(fake_pred, real_label) 
                g_pix_loss = self.pixel_loss_fn(fake_color, color)
                if self.perceptual_loss:
                    g_perc_loss = self.perceptual_loss(fake_color, color)
                else:
                    g_perc_loss = 0.0

                loss_G = 0.01 * g_adv_loss + 1.0 * g_pix_loss + 0.1 * g_perc_loss

                self.g_optimizer.zero_grad()
                loss_G.backward()
                self.g_optimizer.step()

                # ------- Train Discriminator -------
                real_pred = D(sketch, color)   
                real_target = torch.ones_like(real_pred)
                loss_D_real = self.adv_loss_fn(real_pred, real_target)

                fake_pred = D(sketch, fake_color.detach())
                fake_target = torch.zeros_like(fake_pred)
                loss_D_fake = self.adv_loss_fn(fake_pred, fake_target)

                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                self.d_optimizer.zero_grad()
                loss_D.backward()
                self.d_optimizer.step()

            print(f"Epoch {epoch}: Generator loss: {loss_G.item():.4f}, Disc loss: {loss_D.item():.4f}")
            losses_g.append(loss_G.item())
            losses_d.append(loss_D.item())
            os.makedirs(path, exist_ok=True)
            save_val_collage(G, val_loader = validation_loader, device = self.device, save_path=os.path.join(path, f"collage_epoch_{epoch}.png"))
            if epoch > 30:
                self.save_model(G, epoch, path)

        with open(os.path.join(path, "losses.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generator_loss", "discriminator_loss"])  # header

            for g_loss, d_loss in zip(losses_g, losses_d):
                writer.writerow([g_loss, d_loss])
        
