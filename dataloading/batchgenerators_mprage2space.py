#!/usr/bin/env python3
import os
import random

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from batchgenerators.transforms.abstract_transforms import Compose

import numpy as np
from command_line_arguments.command_line_arguments import CommandLineArguments

import classification_config as config


class Mprage2space(DataLoader):

    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         False)

        self.patch_size = patch_size
        self.num_modalities = config.NUM_INPUT_CHANNELS
        self.indices = list(range(len(self._data)))

    def __len__(self):
        return len(self._data)

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for data and seg
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1), dtype="int16")

        for i, j in enumerate(patients_for_batch):
            input_image_path = os.path.join(config.TRAIN_DIR, j)
            input_image = np.load(input_image_path, mmap_mode="r")

            output_image = 1 if j.split("_")[0] == "m" else 0

            # Get tensor from image
            input_image = input_image / config.MAX_VALUE

            data[i] = input_image
            seg[i] = output_image

        return {'data': data, 'class': seg, "sample": j}


def get_train_transform(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=False,
            do_rotation=True,
            do_scale=False,
            random_crop=False,
            p_rot_per_sample=0.66,
        )
    )

    tr_transforms = Compose([])
    return tr_transforms


def get_split():
    random.seed(config.FOLD)

    all_samples = os.listdir(config.TRAIN_DIR)

    percentage_val_samples = 20

    num_val_samples = int(len(all_samples) / 100 * percentage_val_samples)
    val_samples = random.sample(all_samples, num_val_samples)

    train_samples = list(filter(lambda sample: sample not in val_samples, all_samples))

    return train_samples, val_samples


if __name__ == '__main__':

    cma = CommandLineArguments()
    cma.parse_args()

    train_samples, val_samples = get_split()

    dl_train = Mprage2space(train_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                            seed_for_shuffle=config.FOLD,
                            return_incomplete=False, shuffle=True)

    transform = get_train_transform(config.PATCH_SIZE)

    mt_train = SingleThreadedAugmenter(
        data_loader=dl_train,
        transform=Compose([]),
        #num_processes=config.NUM_WORKERS,
    )

    dl_val = Mprage2space(val_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                          return_incomplete=False, shuffle=False)

    mt_val = SingleThreadedAugmenter(
        data_loader=dl_val,
        transform=Compose([]),
        #num_processes=config.NUM_WORKERS,
    )

    for batch in mt_train:
        print(batch["sample"])
