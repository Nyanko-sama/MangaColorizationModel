from datasets import DanbooruDataModule, MangaScanesDataModule
from models import SmallerUnet, ResUnet, FusionNet, GANModel, Generator
from trainers import GanTrainer, GeneratorTrainer
import argparse
import random
import numpy as np

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--fine_epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--split", default=0.05, type=float, help="Validation split for the dataset.")


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

    dan_dataset = DanbooruDataModule("../danbooru", batch_size=args.batch_size, validation_split = args.split, resize=(256, 256))
    dan_small = dan_dataset.smal_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = GeneratorTrainer(
        loss_fn=torch.nn.L1Loss(),
        optimizer=None,
        device=device
    )

    gen = Generator().to(device)
    trainer.optimizer = torch.optim.Adam(gen.parameters(), lr=1e-3)
    trainer.train(gen, dan_small, args.epochs, logs_dir="./logs/new", vis_dir=dan_dataset.path_to_dir)

    gan_trainer = GanTrainer(device=device)
    print('GAN')
    gan_fusion = GANModel(net_G = Generator()).to(device)
    gan_trainer.train_model(gan_fusion, dan_small, args.epochs, logs_dir="./logs/new", vis_dir=dan_dataset.path_to_dir)
    


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)