import nibabel as nib
import os
from torch.utils.data import Dataset
import config
import random
from torch.utils.data import DataLoader
import numpy as np


class SpaceRt(Dataset):
    def __init__(self, root_dir, fold=None, val=False, percentage_val_samples=15):

        if fold is None:
            raise ValueError("No fold given")

        if not fold == "all":
            if not isinstance(fold, int):
                raise ValueError("Fold must be int or 'all'")

        random.seed(fold)

        self.root_dir = root_dir
        self.all_samples = os.listdir(self.root_dir)

        # 15% val. data

        num_val_samples = len(self.all_samples) // 100 * percentage_val_samples
        self.val_samples = random.sample(self.all_samples, num_val_samples)

        self.train_samples = list(filter(lambda sample: sample not in self.val_samples, self.all_samples))

        if val:
            self.list_samples = self.val_samples
        else:
            self.list_samples = self.train_samples

        if fold == "all":
            self.list_samples = self.all_samples

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, index):

        sample = self.list_samples[index]

        # Get image path

        input_image_path = os.path.join(config.TRAIN_DIR, sample,
                                        "mprage_0.nii.gz")
        output_image_path = os.path.join(config.TRAIN_DIR, sample,
                                         "space_0.nii.gz")

        # Load image
        input_image = nib.load(input_image_path)
        output_image = nib.load(output_image_path)

        # Get tensor from image
        input_image = input_image.get_fdata()
        output_image = output_image.get_fdata()

        output_image_max_value = config.MAX_VALUE_SPACE
        input_image_max_value = config.MAX_VALUE_MPRAGE

        # normalize
        input_image = input_image / input_image_max_value  # / 5154.285
        output_image = output_image / output_image_max_value  # / 1435.0

        return input_image, output_image


if __name__ == '__main__':
    train_dataset = SpaceRt(root_dir=config.TRAIN_DIR, fold=0, val=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    for batch in train_loader:
        print(batch)
