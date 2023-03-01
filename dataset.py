import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob

from dgl.geometry import farthest_point_sampler


def load_dataset(config, mode):
    if config.dataset == "shapenet":
        dataset = ShapeNet(config.data_dir, mode, config.num_samples)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        return dataset, dataloader
    
    else:
        raise ValueError()


# iterable dataset
class ShapeNet(Dataset):
    def __init__(self, root, mode, num_samples=None):
        super().__init__()
        self.root = root
        self.num_samples = num_samples

        self.directories = []
        for dir in glob.glob(f"{root}/*"):
            with open(f"{dir}/{mode}.lst") as file:
                for line in file:
                    self.directories.append(f"{dir}/{line.strip()}")

    def __len__(self):
        """return directories size"""
        return len(self.directories)
    
    def __getitem__(self, index):
        """return specify object, point cloud & image"""
        dir = self.directories[index]
        data = np.load(f"{dir}/points.npz")
        positions = data["points"].astype(np.float32)
        occupancies = np.unpackbits(data["occupancies"])[:positions.shape[0]].astype(np.float32)
        image = Image.open(np.random.choice(glob.glob(f"{dir}/img_choy2016/*.jpg"))).convert('RGB')
        image = np.array(image, dtype=np.float32)

        if self.num_samples:
            half_samples = self.num_samples // 2

            # near surface region samples
            near_surface_indices = np.nonzero(occupancies.astype(np.int32))[0]
            if near_surface_indices.shape[0] <= half_samples:
                indices_of_near_surface_indices = np.random.choice(near_surface_indices.shape[0], half_samples)
            else:
                surface_positions = torch.tensor(positions[near_surface_indices])
                surface_positions = surface_positions.unsqueeze(0)
                indices_of_near_surface_indices = farthest_point_sampler(surface_positions, half_samples).numpy()[0]
                
            near_surface_indices = near_surface_indices[indices_of_near_surface_indices]

            # random samples
            random_indices = np.random.choice(positions.shape[0], half_samples, replace=False)

            # integrate indices
            indices = np.concatenate([near_surface_indices, random_indices], axis=0)
            indices = np.random.permutation(indices)

            positions = positions[indices]
            occupancies = occupancies[indices]
        
        positions = np.transpose(positions)
        occupancies = np.transpose(occupancies)
        image = np.transpose(image, [2, 0, 1])

        return positions, occupancies, image
