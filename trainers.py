import torch
from tqdm import tqdm
from datasets import TestVis
import os
import csv

class GeneratorTrainer:
    def __init__(self, optimizer, loss_fn, device):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def evaluate_on_val(self, model, val_loader):
        model.eval()
        losses = 0
        with torch.no_grad():
            for gray, color in val_loader:
                gray = gray.to(self.device)
                color = color.to(self.device)

                # Forward pass
                output = model(gray)

                # Compute loss
                loss = self.loss_fn(output, color)
                losses += loss.item()
        return losses / len(val_loader)

    def train(self, model, train_loader, epochs, logs_dir, vis_dir):
        visualizer = TestVis(vis_dir, resize_shape=(256, 256))
        model.train()
        train_losses = []
        for epoch in range(epochs):
            model.to(self.device)
            losses = 0
            for gray, color in tqdm(train_loader):
                gray = gray.to(self.device)
                color = color.to(self.device)

                # Forward pass
                output = model(gray)

                # Compute loss
                loss = self.loss_fn(output, color)
                losses += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            losses /= len(train_loader)
            
            # Save the images to logs_dir
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {losses:.4f}")
            os.makedirs(logs_dir, exist_ok=True)
            visualizer.visualize(model, path=os.path.join(logs_dir, f'test_{epoch+1}.png'))

            train_losses.append(losses)

        with open(os.path.join(logs_dir, 'training_losses.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'loss'])  # header
            for epoch, train_loss in enumerate(train_losses):
                writer.writerow([epoch + 1, train_loss])


class GanTrainer:
    def __init__(self, device):
        self.device = device
    
    # from https://github.com/mberkay0/image-colorization
    def update_losses(self, model, loss_meter_dict, count):
        for loss_name, loss_meter in loss_meter_dict.items():
            loss = getattr(model, loss_name)
            loss_meter.update(loss.item(), count=count)
    
    # from https://github.com/mberkay0/image-colorization
    def train_model(self, model, train_dl, epochs, vis_dir, logs_dir):
        model.to(self.device)
        visualizer = TestVis(vis_dir, resize_shape=(256, 256))

        for e in range(epochs):
            loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
            i = 0                                  # log the losses of the complete network
            for data in tqdm(train_dl):
                model.setup_input(data) 
                model.optimize()
                self.update_losses(model, loss_meter_dict, count=data[0].size(0)) # function updating the log objects
                i += 1
            os.makedirs(logs_dir, exist_ok=True)
            print(f"\nEpoch {e+1}/{epochs}, Losses: ") 
            log_results(loss_meter_dict, logs_dir) # function to print out the losses
            visualizer.visualize(model.net_G, path=os.path.join(logs_dir, f'test_{e+1}.png'))
        return model

# from https://github.com/mberkay0/image-colorization
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def log_results(loss_meter_dict, logs_dir):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
    
    csv_file = os.path.join(logs_dir, f'training_losses.csv')
    if not os.path.exists(csv_file):
        with open(os.path.join(logs_dir, 'training_losses.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(loss_meter_dict.keys())  # header

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loss_meter_dict.values())  # write the values