import os
import shutil
import random
import torch
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
import numpy as np

def split_data(img_directory, mask_directory, output_directory):
    """
        Splits images + masks into train/val/test folders.
    """
    global img_dir
    global mask_dir
    global output_dir
    img_dir = img_directory
    mask_dir = mask_directory
    output_dir = output_directory
    splits = ["test", "val","train"]

    for s in splits:
        os.makedirs(f"{output_dir}/{s}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/{s}/masks", exist_ok=True)

    train_ratio = 0.70
    val_ratio = 0.15

    #matches patient id to mask file name
    #mask_dict = {m.split("_")[1]:m for m in os.listdir(mask_dir) if not m.startswith(".")}

    pairs = []

    #verifies all images have a corresponding mask, and pairs img and mask
    for img in os.listdir(img_dir):
        if img.startswith('.'):  # skip hidden files
            continue
        
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

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")


def copy_split(pairs_list, split):
        """
        This function takes a paired list of images and masks, and splits them into training, validating, and testing datasets.
        Args:
            pairs_list (tuple): a tuple of the form [img, mask]
            split (string): name of dataset the input should be placed into
        """
        for img, mask in pairs_list:
            shutil.copy(os.path.join(img_dir,img), f"{output_dir}/{split}/images/{img}")
            shutil.copy(os.path.join(mask_dir,mask), f"{output_dir}/{split}/masks/{mask}")


class SegDataset(Dataset):
    """
    Loads image-mask pairs from selected folder
    """

    def __init__(self, img_dir, mask_dir, transform=None):
          self.img_dir = img_dir
          self.mask_dir = mask_dir
          self.transform = transform

          #writing down all file names to know what data we have
          self.images = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
          self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

          self.index_map = []

          self.img_to_mask = {}
          
          for img_file in self.images:
            patient_id = img_file.split('_')[1] # adjust to your naming
            # find matching mask
            #mask_matches = [m for m in os.listdir(mask_dir) if patient_id in m]
            matched_mask = next((m for m in self.masks if patient_id in m), None)
            if matched_mask is None:
                raise ValueError(f"No mask found for {img_file}")
            self.img_to_mask[img_file] = matched_mask

          for vol_idx, img_file in enumerate(self.images):
               img_path = os.path.join(self.img_dir, img_file)
               mask_file = self.img_to_mask[img_file]
               mask_path = os.path.join(self.mask_dir, mask_file)
               
               vol_img = nib.load(img_path).get_fdata()
               vol_mask = nib.load(mask_path).get_fdata()
               depth = vol_img.shape[2]
               
               for slice_idx in range(depth):
                    mask_slice = vol_mask[:, :, slice_idx]
                    if mask_slice.max() > 0:  # only keep slices with foreground
                        self.index_map.append((vol_idx, slice_idx))
    
    #gets number of images in dataset 
    def __len__(self):
         return len(self.index_map)
    
    def __getitem__(self, idx):
         vol_idx, slice_idx = self.index_map[idx]
         
         img_file = self.images[vol_idx]
         mask_file = self.img_to_mask[img_file] ##make them have same name, but just in different directory

         img_path = os.path.join(self.img_dir,img_file)
         mask_path = os.path.join(self.mask_dir, mask_file)

         #loading image and mask

         img = nib.load(img_path).get_fdata()
         mask = nib.load(mask_path).get_fdata()

         img_slice = img[:,:, slice_idx]
         mask_slice = mask[:,:,slice_idx]

         mask_slice = np.squeeze(mask_slice)

         if mask_slice.ndim != 2:
            raise RuntimeError(f"Mask slice not 2D, got shape {mask_slice.shape}")
        
         #converting to torch tensor
         img_slice = torch.tensor(img_slice).unsqueeze(0).float()
         mask_slice = torch.tensor(mask_slice).float()

         if self.transform:
              img_slice = self.transform(img_slice)

         #mim-max normalization image slice
         img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
         #fit max slice to correct size
         mask_slice = F.interpolate(mask_slice.unsqueeze(0).unsqueeze(0).float(), 
                           size=(256, 256), mode='nearest').squeeze(0).squeeze(0).long()
        
         
         return img_slice, mask_slice

