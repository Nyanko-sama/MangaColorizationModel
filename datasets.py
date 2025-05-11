import matplotlib.pyplot as plt
import torch
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from scipy import stats

import torchvision.transforms as transforms

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
        self.resize = resize
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
        
class SketchDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for the MangaScanes dataset.
    Returns (L channel, ab channels) tuples.
    """
    def __init__(self, image_paths, resize = (256, 256), out_ch = 2):
        self.image_paths = image_paths
        self.pad = NewPad()
        self.resize = resize
        self.out_ch = out_ch

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        color = img[:, :512, :]     # Left half → [C, 512, 512]
        line = img[:, 512:, :]    # Right half → [C, 512, 512]

        # resize with padding to keep aspect ratio
        color = cv2.resize(color, self.resize)
        line = cv2.resize(line, self.resize)
        if self.out_ch == 2:
            color = cv2.cvtColor(color, cv2.COLOR_RGB2LAB)

            L, A, B = cv2.split(color)

            L = (L / 50) - 1
            A = (A - 128) / 128.0
            B = (B - 128) / 128.0
            gray = torch.tensor(L, dtype=torch.float32).unsqueeze(0)
            color = torch.tensor(np.stack([A, B], axis=0), dtype=torch.float32)
        else:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            gray = (gray / 255.0) * 2 - 1
            color = (color.astype(np.float32) / 255.0) * 2 - 1
            gray = torch.from_numpy(gray).unsqueeze(0).float()
            color = torch.from_numpy(color)
            color = color.permute(2, 0, 1).contiguous().float() # [C, H, W]
            color = torch.clamp(color, -1.0, 1.0)
            assert torch.isfinite(color).all()
            assert color.min() >= -1.0 and color.max() <= 1.0, f"color range is off: {color.min()} to {color.max()}"

        return gray, color # input (1C), target (2C)


class SketchDataModule:
    """
    Prepares training and validation DataLoaders from MangaScanes dataset path.
    """
    def __init__(self, path, batch_size=32, resize = (256, 256), out_ch = 2):
        self.path_train = os.path.join(path, 'train')
        self.path_val = os.path.join(path, 'val')
        self.path_test = os.path.join(path, 'test')
        self.path_to_dir = path
        self.batch_size = batch_size
        self.temp_dir = None
        self.resize = resize
        self.out_ch = out_ch

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.prepare_data()

    def prepare_data(self):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        train_images = [os.path.join(root, file)
                      for root, _, files in os.walk(self.path_train)
                      for file in files if file.lower().endswith(extensions)]
        
        val_images = [os.path.join(root, file)
                      for root, _, files in os.walk(self.path_val)
                      for file in files if file.lower().endswith(extensions)]
        
        test_images = [os.path.join(root, file)
                      for root, _, files in os.walk(self.path_test)
                      for file in files if file.lower().endswith(extensions)]

        print(f"Found {len(train_images)} train images and {len(val_images)} val images")

        train_dataset = SketchDataset(train_images, resize=self.resize, out_ch=self.out_ch)
        val_dataset = SketchDataset(val_images, resize=self.resize, out_ch=self.out_ch)
        test_dataset = SketchDataset(test_images, resize=self.resize, out_ch=self.out_ch)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) 
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

