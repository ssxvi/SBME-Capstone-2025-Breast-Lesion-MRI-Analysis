import os
import shutil
import random
import torch
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
# import torchvision.transforms as T
import numpy as np


def split_data(img_dir: str, mask_dir: str, output_dir: str):
    """
        Splits images + masks into train/val/test folders.
    """
    
    splits = ["test", "val","train"]

    for s in splits:
        make_empty_dir(f"{output_dir}/{s}/images")
        make_empty_dir(f"{output_dir}/{s}/masks")

    train_ratio = 0.70
    val_ratio = 0.15

    pairs = []

    #verifies all images have a corresponding mask, and pairs img and mask
    for img in listdir_nohidden(img_dir):
        patient_id = img.split("_")[1]

        mask_matches = [m for m in os.listdir(mask_dir) if patient_id in m]
        if mask_matches:
            pairs.append((img, mask_matches[0]))
        else:
            print(f"WARNING: No mask found for {img}")
        

    #randomizes order so that split has different imgs every time
    random.shuffle(pairs)

    n = len(pairs)

    #defining where train, val, and test datasets end
    train_end = int(n*train_ratio)
    val_end = train_end + int(n*val_ratio)

    train_files = pairs[:train_end]
    val_files = pairs[train_end:val_end]
    test_files = pairs[val_end:]

    copy_split(train_files, "train", img_dir, mask_dir, output_dir)
    copy_split(val_files, "val", img_dir, mask_dir, output_dir)
    copy_split(test_files, "test", img_dir, mask_dir, output_dir)


def copy_split(pairs_list: list[tuple], split: str, img_dir: str, mask_dir: str, output_dir: str):
        """
        This function takes a paired list of images and masks, and splits them into training, validating, and testing datasets.
        Args:
            pairs_list (tuple): a tuple of the form [img, mask]
            split (string): name of dataset the input should be placed into
        """
        images = [x[0] for x in pairs_list]
        masks = [y[1] for y in pairs_list]
        split_directory = f"{output_dir}/{split}"

        copy_files(images, img_dir, f"{split_directory}/images")
        copy_files(masks, mask_dir, f"{split_directory}/masks")

    

def make_empty_dir(directory: str):
    """
    This function creates the desired directory. The newly made directory is ALWAYS empty.
    
    :param directory (string): Name of directory to be created.
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
        return
    
    os.makedirs(directory)


def copy_files(files: str, source: str, destination: str):
    """Uses threads to improve copy speed"""
    n_threads = min(len(files), os.cpu_count())
    with ThreadPoolExecutor(n_threads) as executor:
        _ = [executor.submit(shutil.copy, os.path.join(source, file), f"{destination}/{file}") for file in files]

def listdir_nohidden(path: str):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def map_imgs_to_masks(img_files, mask_files):
    img_to_mask_file_map = {}

    for f in img_files:
        patient_id = f.split('_')[1] #adjust to match naming convention
        mask_file = [m for m in mask_files if patient_id in m]
        if not mask_file:
            raise ValueError(f"No mask found for {f}")
        img_to_mask_file_map[f] = mask_file[0]

    return img_to_mask_file_map


def min_max_normalization(slice):
    return ((slice - slice.min()) / (slice.max() - slice.min() + 1e-6))



class SegDataset(Dataset):
    """
    Loads image-mask pairs from selected folder
    """

    def __init__(self, img_dir, mask_dir):
          #writing down all file names to know what data we have
          self.img_files = sorted(listdir_nohidden(img_dir))
          self.mask_files = sorted(listdir_nohidden(mask_dir))

          #mapping image filenames -> maskfilenames
          self.img_to_mask = map_imgs_to_masks(self.img_files, self.mask_files)

          # storing all images as numpy arrays
          #list of tuples (img_vol, mask_vol) such that img_vol corresponds to mask_vol
          self.volumes = [
              (nib.load(os.path.join(img_dir, f)).get_fdata(),
               nib.load(os.path.join(mask_dir, self.img_to_mask[f])).get_fdata())
               for f in self.img_files
          ]
          
          margin = 5
          self.index_map = []
          #reducing volume to slices that have segmentation_mask
          for vol_idx, (img_vol,mask_vol) in enumerate(self.volumes):
            depth = min(img_vol.shape[2], mask_vol.shape[2])
            fg_slices = [slice_idx for slice_idx in range(depth) if mask_vol[:,:,slice_idx].max() > 0]

            if len(fg_slices) == 0:
                continue

            min_slice = max(0, min(fg_slices) - margin)
            max_slice = min(depth - 1, max(fg_slices) + margin)
            self.index_map.extend([(vol_idx, slice_idx) for slice_idx in range(min_slice, max_slice + 1)])
            # mask_data = mask_vol 
            # self.index_map.extend([(vol_idx,slice_idx) for slice_idx in range(depth) if mask_data[:,:,slice_idx].max() > 0])
          #temporarily training on one image only
        #   self.volumes = self.volumes[:1]
          
    #gets number of slices in dataset 
    def __len__(self):
         return len(self.index_map)
    
    def __getitem__(self, idx):
         vol_idx, slice_idx = self.index_map[idx]
         img_slice = self.volumes[vol_idx][0][:,:,slice_idx]
         mask_slice = self.volumes[vol_idx][1][:,:,slice_idx]

         #ensuring slices are 2D not 3D or other dimension
         mask_slice = np.squeeze(mask_slice)
         if mask_slice.ndim != 2:
            raise RuntimeError(f"Mask slice not 2D, got shape {mask_slice.shape}")
         
         #convert mask and img to torch tensor: 1 x H x W
         img_slice = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
         mask_slice = torch.tensor(mask_slice, dtype=torch.long).unsqueeze(0)

         #resize images to 1 x 256 x 256
         img_slice = F.interpolate(img_slice.unsqueeze(0), size=(256,256), mode = 'bilinear', align_corners=False).squeeze(0)
         mask_slice = F.interpolate(mask_slice.unsqueeze(0).float(), size=(256,256), mode ='nearest').squeeze(0).long()
        
         p1 = torch.quantile(img_slice, 0.01)
         p99 = torch.quantile(img_slice, 0.99)
         img_slice = torch.clamp(img_slice, p1, p99)
         img_slice = (img_slice - p1) / (p99 - p1 + 1e-6)

         
         #image pre_processing: mim-max normalization
        #  img_slice = min_max_normalization(img_slice)

         return img_slice, mask_slice

