import torch
from torch import utils
from PIL import Image
import numpy as np
import glob


# iterable dataset
class ShapeNet(utils.data.Dataset):
    def __init__(self, root, mode, num_samples=None):
        super().__init__()
        self.root = root
        self.num_samples = num_samples

        self.directories = []
        for dir in glob.glob(f"{root}/*"):
            with open(f"{dir}/{mode}.lst") as file:
                for line in file:
                    self.directories.append(f"{dir}/{line}")

    def __len__(self):
        """return directories size"""
        return len(self.directories)
    
    def __getitem__(self, index):
        """return specify object, point cloud & image"""
        dir = self.directories[index]
        data = np.load(f"{dir}/points.npz")
        positions = data["points"].astype(np.float32)
        occupancies = np.unpackbits(data["occupancies"])[:positions.shape[0]].astype(np.float32)
        image = Image.open(np.random.choice(glob.glob(f"{dir}/img_choy2016/*.jpg")))

        if self.num_samples:
            indices = np.random.choice(positions.shape[0], self.num_samples, replace=False)
            positions = positions[indices]
            occupancies = occupancies[indices]
        
        positions = np.transpose(positions)
        occupancies = np.transpose(occupancies)

        return positions, occupancies, image
