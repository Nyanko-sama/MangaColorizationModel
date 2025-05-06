import matplotlib.pyplot as plt
import torch
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

#####################   DanbooruDataset #####################
class DanbooruDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the Danbooru dataset.
    Returns (L channel, ab channels) tuples.
    """
    def __init__(self, image_paths, resize = (512, 512)):
        self.image_paths = image_paths
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, self.resize)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(img)

        A = (A - 128) / 128.0
        B = (B - 128) / 128.0
        L = (L / 50) - 1
        gray = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
        color = torch.tensor(np.stack([A, B], axis=0), dtype=torch.float32)

        return gray, color  # input (1C), target(2C)


class DanbooruDataModule:
    """
    Prepares training and validation DataLoaders from Danbooru dataset path.
    """
    def __init__(self, path, batch_size=32, validation_split=0.05, resize = (512, 512)):
        self.path = os.path.join(path, 'train')
        self.path_to_dir = path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.temp_dir = None
        self.resize = resize

        self.train_loader = None
        self.val_loader = None
        self.prepare_data()

    def prepare_data(self):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        all_images = [os.path.join(root, file)
                      for root, _, files in os.walk(self.path)
                      for file in files if file.lower().endswith(extensions)]

        print(f"Found {len(all_images)} images")

        _, small = train_test_split(all_images, test_size=self.validation_split)

        train_dataset = DanbooruDataset(all_images, resize=self.resize)
        small_dataset = DanbooruDataset(small, resize=self.resize)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.smal_loader = torch.utils.data.DataLoader(small_dataset, batch_size=self.batch_size, shuffle=True)

#####################   MangaScanesDataset  #####################
# inspired by https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/3 
class NewPad(object):
    def __init__(self, fill=255.0):
        self.fill = fill
        
    def __call__(self, img):
        """
        Args: img (PIL Image): Image to be padded.
        Returns: PIL Image: Padded image.
        """
        padding = self.get_padding(img)
        return cv2.copyMakeBorder(
            img,
            top=padding[0],
            bottom=padding[1],
            left=padding[2],
            right=padding[3],
            borderType=cv2.BORDER_CONSTANT,
            value=self.fill
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(fill={self.fill}, padding_mode='constant')"

    def  get_padding(self, image):    
        h, w = image.shape[:2]
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return (int(pad_top), int(pad_bottom), int(pad_left), int(pad_right))
    
class MangaScanesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the MangaScanes dataset.
    Returns (L channel, ab channels) tuples.
    """
    def __init__(self, image_paths, resize = (512, 512)):
        self.image_paths = image_paths
        self.pad = NewPad()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize with padding to keep aspect ratio
        img = self.pad(img)
        img = cv2.resize(img, self.resize)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(img)

        L = (L / 50) - 1
        A = (A - 128) / 128.0
        B = (B - 128) / 128.0
        gray = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
        color = torch.tensor(np.stack([A, B], axis=0), dtype=torch.float32)

        return gray, color  # input (1C), target (2C)


class MangaScanesDataModule:
    """
    Prepares training and validation DataLoaders from MangaScanes dataset path.
    """
    def __init__(self, path, batch_size=32, validation_split=0.1, resize = (512, 512)):
        self.path = os.path.join(path, 'train')
        self.path_to_dir = path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.temp_dir = None
        self.resize = resize

        self.train_loader = None
        self.smal_loader = None
        self.prepare_data()

    def prepare_data(self):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        all_images = [os.path.join(root, file)
                      for root, _, files in os.walk(self.path)
                      for file in files if file.lower().endswith(extensions)]

        print(f"Found {len(all_images)} images")

        _, small = train_test_split(all_images, test_size=self.validation_split)

        train_dataset = MangaScanesDataset(all_images, resize=self.resize)
        small_dataset = MangaScanesDataset(small, resize=self.resize)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.smal_loader = torch.utils.data.DataLoader(small_dataset, batch_size=self.batch_size, shuffle=True) 
            
################### TestVis #######################
class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, resize_shape = (512, 512)):
        self.image_paths = image_paths
        self.pad = NewPad()
        self.resize = resize_shape

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize with padding to keep aspect ratio
        img = self.pad(img)
        img = cv2.resize(img, self.resize)

        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(img_lab)

        L = (L / 50) - 1
        A = (A - 128) / 128.0
        B = (B - 128) / 128.0
        gray = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
        color = torch.tensor(np.stack([A, B], axis=0), dtype=torch.float32)

        return gray, color

class TestVis:
    """
    Loads test images, applies the model, and visualizes predictions.
    """
    def __init__(self, path, resize_shape=(512, 512)):
        self.device = torch.device("cpu")
        self.image_paths = self._load_image_paths(os.path.join(path, 'test'))
        self.dataset = TestImageDataset(self.image_paths, resize_shape=resize_shape)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=5, shuffle=False)

    def _load_image_paths(self, path):
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        return [
            os.path.join(root, f)
            for root, _, files in os.walk(path)
            for f in files if f.lower().endswith(valid_exts)
        ]

    def visualize(self, model, path='test_results.png'):
        model.eval()
        model.to(self.device)

        batch = next(iter(self.dataloader))  # Get up to n images
        inputs, _ = batch
        inputs = inputs.to(self.device)
        with torch.no_grad():
            predictions = model(inputs)

        # Convert to CPU for plotting
        originals = inputs.cpu().numpy()
        preds = predictions.cpu().numpy()

        imgs_to_plot = []
        for i in range(preds.shape[0]):
            L = (inputs[i] + 1) * 50.0 
            L = L.squeeze()
            AB = preds[i]  
            A = (AB[0] * 128.0 + 128)
            B = (AB[1] * 128.0 + 128)

            lab = np.stack([L, A, B], axis=-1).astype(np.uint8)  # (H, W, 3)
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            imgs_to_plot.append(rgb)
        
        preds = imgs_to_plot

        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(originals[i].squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Original')

            plt.subplot(2, 5, i + 5 + 1)
            output_img = preds[i] 
            output_img = output_img.clip(0, 1)
            if output_img.shape[2] == 1:
                output_img = output_img.squeeze()
                plt.imshow(output_img, cmap='gray')
            else:
                plt.imshow(output_img)
            plt.axis('off')
            plt.title('Predicted')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved prediction visualization to {path}")
