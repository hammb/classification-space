#!/usr/bin/env python3
import copy
import os
import random
import time

import numpy as np
import torchvision.models as models
import torchvision.utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import classification_config as config
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from dataloading.batchgenerators_mprage2space import Mprage2space
import torch
from command_line_arguments.command_line_arguments import CommandLineArguments


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
                                               p_per_channel=0.5, p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)

    return tr_transforms


def get_split():
    random.seed(config.FOLD)

    all_samples = os.listdir(config.TRAIN_DIR)

    percentage_val_samples = 15
    # 15% val. data

    num_val_samples = int(len(all_samples) / 100 * percentage_val_samples)
    val_samples = random.sample(all_samples, num_val_samples)

    train_samples = list(filter(lambda sample: sample not in val_samples, all_samples))

    return train_samples, val_samples


def train_fn(model, criterion, mt_train, optimizer, epoch):
    # loop = tqdm(mt_train, total=len(mt_train.data_loader.indices), leave=True)
    # loop.set_description("Training Epoch Nr.: " + str(epoch))
    # random.seed(epoch)
    # losses_batches = []
    for batch_idx, batch in enumerate(mt_train):
        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["class"]).to(config.DEVICE)

        #img_grid = torchvision.utils.make_grid(x[:,random.randint(0,2),8,:,:])
        #writer.add_image("batchgenerators", img_grid)

        with torch.cuda.amp.autocast():
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs.type(torch.float32), y.type(torch.float32))
            loss.backward()
            optimizer.step()
            # losses_batches.append(loss.item())

    return loss.item()


def evaluate(model, epoch, fold, mt_val, train_loss):
    model.eval()
    all_outputs = torch.Tensor()
    all_y = torch.Tensor()
    corrects = []
    losses_batches = []
    for batch_idx, batch in enumerate(mt_val):
        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["class"]).to(config.DEVICE)

        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs.type(torch.float32), y.type(torch.float32))

            corrects.append(torch.sum(torch.round(torch.sigmoid(outputs)) == y).detach().to('cpu').numpy().min())
            all_outputs = torch.cat([all_outputs, outputs.detach().to('cpu')])
            all_y = torch.cat([all_y, y.detach().to('cpu')])
            losses_batches.append(loss.item())

    print('[%d] Corr.: %d/%d, T-Loss: %.3f, V-Loss: %.3f' % (epoch + 1, all_y.shape[0], np.sum(corrects), train_loss, np.mean(losses_batches)))

    model.train()

    return np.mean(losses_batches), all_y.shape[0]/np.sum(corrects)


if __name__ == '__main__':

    cma = CommandLineArguments()
    cma.parse_args()

    config.CHECKPOINT_PATH = os.path.join(os.environ['cs_checkpoint_path'], os.path.basename(__file__).split(".")[0])
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    config.CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_PATH, config.TASK)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD), exist_ok=True)

    num_classes = 1

    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    model.stem[0] = nn.Conv3d(config.NUM_INPUT_CHANNELS, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(config.NUM_INPUT_CHANNELS, 3, 3), bias=False)
    model = model.cuda()

    train_samples, val_samples = get_split()

    dl_train = Mprage2space(train_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                            seed_for_shuffle=config.FOLD,
                            return_incomplete=False, shuffle=True)

    transform = get_train_transform(config.PATCH_SIZE)

    mt_train = MultiThreadedAugmenter(
        data_loader=dl_train,
        transform=transform,
        num_processes=config.NUM_WORKERS,
    )

    dl_val = Mprage2space(val_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                          return_incomplete=False, shuffle=False)

    mt_val = MultiThreadedAugmenter(
        data_loader=dl_val,
        transform=Compose([]),
        num_processes=config.NUM_WORKERS,
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    if config.TENSOR_BOARD:
        writer = SummaryWriter(os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD, "runs/classification/track_stats"))

    model.train()
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    best_corrects = 10
    best_val_loss = 10
    best_train_loss = 10
    for epoch in range(0, config.NUM_EPOCHS):

        train_loss = train_fn(
            model, criterion, mt_train, optimizer, epoch
        )

        val_loss, corrects = evaluate(model, epoch, config.FOLD, mt_val, train_loss)

        if config.TENSOR_BOARD:
            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('val_loss', val_loss, global_step=epoch)
            writer.add_scalar('val_acc', corrects, global_step=epoch)

        if val_loss < best_val_loss:

            best_corrects = corrects
            best_val_loss = val_loss
            best_train_loss = train_loss

            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD,
                                                           "../model_best.pth.tar"))

    print('Finished Training')
    save_checkpoint(model, optimizer, os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD, "model_end.pth.tar"))
    model.load_state_dict(best_model_wts)
