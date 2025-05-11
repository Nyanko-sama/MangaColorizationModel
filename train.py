from datasets import DanbooruDataModule, MangaScanesDataModule, SketchDataModule
import argparse
import random
import numpy as np

from models import ColorizationUNet, PatchDiscriminator
from utils import Trainer

import torch
from loss import PerceptualLoss, AnimeLoss

import os

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--fine_epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For single GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU (if used)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args: argparse.Namespace) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sketch_dataset = SketchDataModule("./images", batch_size=args.batch_size, resize=(256, 256), out_ch=3)
    sketch_loader = sketch_dataset.train_loader
    val_loader = sketch_dataset.val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print('Perceptual loss')
    g_loss = PerceptualLoss()
    print('Attention, no extractor, perceptual loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/att_perceptual")

    print('Attention, extractor, perceptual loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=True)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/att_ext_perceptual")

    print('No attention, extractor, perceptual loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/ext_perceptual")

    print('No attention, no extractor, perceptual loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=False, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/perceptual")



    print('ANIME loss')
    g_loss = AnimeLoss()
    print('Attention, no extractor, anime loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/att_anime")

    print('Attention, extractor, anime loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=True)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/att_ext_anime")

    print('No attention, extractor, anime loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/ext_anime")

    print('No attention, no extractor, anime loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=False, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/anime")


    print('L1 Loss')
    g_loss = None
    print('Attention, no extractor, l1 loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/att_l1")

    print('Attention, extractor, l1 loss') # Attention, extractor, l1 loss
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=True)
    G.to(device)
    G.load_state_dict(torch.load("./logs/att_ext_l1/model_epoch_44.pth"))
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=5, path="./logs/att_ext_l1")
    torch.save(G.state_dict(), os.path.join("./logs/att_ext_l1", f"model_epoch_49.pth"))

    print('No attention, extractor, l1 loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=True, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/ext_l1")

    print('No attention, no extractor, l1 loss')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=False, use_extractor=False)
    G.to(device)
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/l1")


    g_loss = AnimeLoss(a = 1)
    print('No attention, no extractor, anime loss 100%')
    G = ColorizationUNet(in_channels=1, out_channels=3, use_attention=False, use_extractor=False)
    G.to(device)
    G.load_state_dict(torch.load("./logs/anime_100/model_epoch_34.pth"))
    D = PatchDiscriminator()
    D.to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    trainer = Trainer(g_optimizer, d_optimizer, g_loss=g_loss, device=device)
    trainer.train(G, D, sketch_loader, val_loader, num_epochs=args.epochs, path="./logs/anime_100")
    torch.save(G.state_dict(), os.path.join("./logs/anime_100", f"model_epoch_49.pth"))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)