import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import classification_config as config


class ClaSpace(Dataset):
    def __init__(self, root_dir, list_samples=None):
        self.root_dir = root_dir
        if list_samples is None:
            self.list_samples = os.listdir(self.root_dir)
        else:
            self.list_samples = list_samples

    def __len__(self):
        return len(self.list_samples)

    def __getitem__(self, index):
        sample = self.list_samples[index]

        input_image_path = os.path.join(config.TRAIN_DIR, sample)
        input_image = np.load(input_image_path, mmap_mode="r")

        output_image = 1 if sample.split("_")[0] == "m" else 0

        if output_image:
            output_image = torch.ones(1)[None, :]
        else:
            output_image = torch.zeros(1)[None, :]

        # Get tensor from image
        input_image = input_image / config.MAX_VALUE

        return torch.Tensor(input_image).float(), output_image


# train_dataset = ClaSpace(root_dir=config.TRAIN_DIR)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=128,
#     shuffle=True,
#     num_workers=config.NUM_WORKERS
# )

# for batch in train_loader:
#     print(batch[0].size(), batch[1].size())
