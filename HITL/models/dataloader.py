import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CudaDataLoader:
    def __init__(self, x_tensor, y_tensor=None, batch_size=None, sampler_weights=None, shuffle=True):
        """
        CudaDataLoader for efficient GPU-based data loading.
        
        Args:
            x_tensor: Input tensor (required)
            y_tensor: Target tensor (optional, for training). If None, only yields x_tensor (for prediction)
            batch_size: Batch size (required if y_tensor is provided, optional if y_tensor is None)
            sampler_weights: Sample weights for weighted sampling (optional)
            shuffle: Whether to shuffle data (default: True, ignored if y_tensor is None)
        """
        self.x = x_tensor
        self.y = y_tensor
        self.batch_size = batch_size if batch_size is not None else x_tensor.size(0)
        # Ensure weights are on the same device as x_tensor and properly normalized
        if sampler_weights is not None:
            self.weights = sampler_weights.to(x_tensor.device)
            # Normalize weights to sum to 1 for numerical stability (torch.multinomial does this internally, but explicit normalization is safer)
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = None
        self.shuffle = shuffle
        self.n_samples = x_tensor.size(0)
        self.prediction_mode = y_tensor is None

    def __iter__(self):
        # PREDICTION MODE: Only yield x_tensor (no labels)
        if self.prediction_mode:
            for i in range(0, self.n_samples, self.batch_size):
                yield self.x[i : i + self.batch_size]
            return
        
        # TRAINING MODE: Yield (x_tensor, y_tensor) tuples
        # STRATEGY A: Weighted Sampling (If weights provided)
        if self.weights is not None:
            # We determine how many batches to run. 
            # Usually we run len(data) / batch_size steps, 
            # even though we are oversampling the minority class.
            num_batches = (self.n_samples + self.batch_size - 1) // self.batch_size
            
            for _ in range(num_batches):
                # torch.multinomial is the GPU equivalent of WeightedRandomSampler
                # replacement=True is crucial for oversampling minority classes
                indices = torch.multinomial(self.weights, self.batch_size, replacement=True)
                yield self.x[indices], self.y[indices]
                
        # STRATEGY B: Standard Shuffle (If no weights)
        elif self.shuffle:
            perm = torch.randperm(self.n_samples, device=self.x.device)
            for i in range(0, self.n_samples, self.batch_size):
                indices = perm[i : i + self.batch_size]
                yield self.x[indices], self.y[indices]
                
        # STRATEGY C: Sequential (Validation/Test)
        else:
            for i in range(0, self.n_samples, self.batch_size):
                yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class RGBImageDataset(Dataset):
    """
    Standard disk-backed RGB image dataset for 128x128 inputs.

    Args:
        image_paths: List of image file paths.
        labels: Optional list/array of labels aligned with image_paths.
        image_size: Target square size (default: 128).
        mean/std: Optional normalization stats in RGB order.
    """

    def __init__(self, image_paths, labels=None, image_size=128, mean=None, std=None):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.image_size = image_size
        self.mean = mean
        self.std = std

        if self.labels is not None and len(self.labels) != len(self.image_paths):
            raise ValueError("labels length must match image_paths length")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Convert BGR -> RGB and resize to network input size.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # HWC uint8 -> CHW float32 in [0, 1].
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img)

        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        if self.labels is None:
            return image

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def build_rgb_dataloader(
    image_paths,
    labels=None,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    image_size=128,
    mean=None,
    std=None,
):
    """
    Build a standard CPU dataloader for RGB images (streamed from disk).
    """
    dataset = RGBImageDataset(
        image_paths=image_paths,
        labels=labels,
        image_size=image_size,
        mean=mean,
        std=std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if labels is not None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if labels is not None else False,
    )