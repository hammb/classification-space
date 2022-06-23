#!/usr/bin/env python3
import copy
import os
import random
import time
from array import array
from enum import Enum

import numpy as np
import torchvision.models as models
import monai
import torchvision.utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import logging
from smpl_models import resnet
import classification_config as config
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform_2,
    MirrorTransform,
    Rot90Transform,
)
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.sample_normalization_transforms import (
    ZeroMeanUnitVarianceTransform,
)

from dataloading.batchgenerators_mprage2space import Mprage2space
import torch
from command_line_arguments.command_line_arguments import CommandLineArguments
from torchmetrics import F1Score
import matplotlib.pyplot as plt

f1_score = F1Score(num_classes=2)

steps_of_val = []
all_f1 = []
all_vloss = []
all_lr = []

all_steps = []
all_tloss = []
epochs_steps = {}
model_best = 0


class ModelChoices(Enum):
    MONAI_RESNET = "monai_resnet"
    MONAI_EFFICIENTNET = "monai_effnet"
    MOANI_DENSENET = "monai_densenet"
    TV_RESNET = "video_resnet"
    SMPL_RESNET_18 = "smpl_resnet_18"
    SMPL_RESNET_50 = "smpl_resnet_50"
    SMPL_RESNET_101 = "smpl_resnet_101"
    SMPL_RESNET_152 = "smpl_resnet_152"
    SMPL_RESNET_200 = "smpl_resnet_200"


def get_model(model_name):
    if model_name == "monai_densenet":
        model = monai.networks.nets.DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=1
        )
        model.features[0] = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

    elif model_name == "monai_effnet":
        model = monai.networks.nets.EfficientNetBN(
            "efficientnet-b0",
            pretrained=True,
            spatial_dims=3,
            in_channels=1,
            num_classes=1,
        )
        model._conv_stem = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            32,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )
        model._fc = nn.Linear(1280, num_classes, bias=True)
    elif model_name == "monai_resnet":
        model = monai.networks.nets.resnet18(
            pretrained=False, spatial_dims=3, no_max_pool=True
        )

        if config.PRETRAINED:
            model.conv1 = nn.Conv3d(
                config.NUM_INPUT_CHANNELS,
                64,
                kernel_size=(7, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            )

            checkpoint = torch.load(
                config.PATH_TO_MONAI_WEIGHTS, map_location=config.DEVICE
            )
            checkpoint_state_dict = checkpoint["state_dict"]
            new_checkpoint_state_dict = {}
            for key in checkpoint_state_dict:
                new_checkpoint_state_dict[key[7:]] = checkpoint_state_dict[key]

            model.load_state_dict(new_checkpoint_state_dict, strict=False)

        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(512, num_classes, bias=True)
    elif model_name == "video_resnet":
        model = models.video.r3d_18(pretrained=True)
        model.fc = nn.Linear(512, num_classes, bias=True)
        model.stem[0] = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
    elif model_name == "smpl_resnet_18":
        model = resnet.generate_model(
            model_depth=18,
            n_classes=num_classes,
            n_input_channels=config.NUM_INPUT_CHANNELS,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=3,
            no_max_pool=True,
            widen_factor=1.0,
        )
        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(512, num_classes, bias=True)
    elif model_name == "smpl_resnet_50":
        model = resnet.generate_model(
            model_depth=50,
            n_classes=num_classes,
            n_input_channels=config.NUM_INPUT_CHANNELS,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=3,
            no_max_pool=True,
            widen_factor=1.0,
        )
        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(2048, num_classes, bias=True)

    elif model_name == "smpl_resnet_101":
        model = resnet.generate_model(
            model_depth=101,
            n_classes=num_classes,
            n_input_channels=config.NUM_INPUT_CHANNELS,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=3,
            no_max_pool=True,
            widen_factor=1.0,
        )
        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(2048, num_classes, bias=True)
    elif model_name == "smpl_resnet_152":
        model = resnet.generate_model(
            model_depth=152,
            n_classes=num_classes,
            n_input_channels=config.NUM_INPUT_CHANNELS,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=3,
            no_max_pool=True,
            widen_factor=1.0,
        )
        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(2048, num_classes, bias=True)
    elif model_name == "smpl_resnet_200":
        model = resnet.generate_model(
            model_depth=200,
            n_classes=num_classes,
            n_input_channels=config.NUM_INPUT_CHANNELS,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=3,
            no_max_pool=True,
            widen_factor=1.0,
        )
        model.conv1 = nn.Conv3d(
            config.NUM_INPUT_CHANNELS,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(2048, num_classes, bias=True)
    else:
        raise NotImplementedError("whoops")

    return model


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
    model.load_state_dict(checkpoint["state_dict"], strict=False)
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

    tr_transforms.append(ZeroMeanUnitVarianceTransform())

    tr_transforms.append(
        Compose(
            [
                SpatialTransform_2(
                    patch_size,
                    [i // 2 for i in patch_size],
                    do_elastic_deform=True,
                    deformation_scale=(0, 0.25),
                    do_rotation=True,
                    angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                    angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                    angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
                    do_scale=True,
                    scale=(0.75, 1.25),
                    border_mode_data="constant",
                    border_cval_data=0,
                    border_mode_seg="constant",
                    border_cval_seg=0,
                    order_seg=1,
                    order_data=3,
                    random_crop=False,  # Make sure we don't crop out the true positive metastasis
                    p_el_per_sample=0.1,
                    p_rot_per_sample=0.1,
                    p_scale_per_sample=0.1,
                ),
                Rot90Transform(num_rot=(1,), axes=(1, 2), p_per_sample=0.3),
            ]
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    tr_transforms.append(
        BrightnessMultiplicativeTransform(
            (0.7, 1.5), per_channel=True, p_per_sample=0.15
        )
    )

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(
        GammaTransform(
            gamma_range=(0.5, 2),
            invert_image=False,
            per_channel=True,
            p_per_sample=0.15,
        )
    )
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(
        GammaTransform(
            gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15
        )
    )

    # Gaussian Noise
    tr_transforms.append(
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)
    )

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    tr_transforms.append(
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.5),
            different_sigma_per_channel=True,
            p_per_channel=0.5,
            p_per_sample=0.15,
        )
    )

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


step = 0


def train_fn(model, criterion, mt_train, optimizer, scheduler, epoch):
    # loop = tqdm(mt_train, total=len(mt_train.data_loader.indices), leave=True)
    # loop.set_description("Training Epoch Nr.: " + str(epoch))
    # random.seed(epoch)

    global step
    losses_batches = []
    for batch_idx, batch in enumerate(mt_train):
        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["class"]).to(config.DEVICE)

        # img_grid = torchvision.utils.make_grid(x[:, :, 8, :, :])
        # writer.add_image("batchgenerators", img_grid)

        with torch.cuda.amp.autocast():
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs.type(torch.float32), y.type(torch.float32))
            loss.backward()
            optimizer.step()
            detached_loss = loss.detach().cpu().numpy()
            losses_batches.append(detached_loss)
        step = step + 1
        all_steps.append(step)
        all_tloss.append(detached_loss)
    if epoch > config.SCHEDULER_ENTRY:
        scheduler.step()

    return np.mean(losses_batches)


def evaluate(model, epoch, mt_val, train_loss):
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

            corrects.append(
                torch.sum(torch.round(torch.sigmoid(outputs)) == y)
                .detach()
                .to("cpu")
                .numpy()
                .min()
            )
            all_outputs = torch.cat([all_outputs, outputs.detach().to("cpu")])
            all_y = torch.cat([all_y, y.detach().to("cpu")])
            losses_batches.append(loss.item())

    f_1 = f1_score(
        torch.round(torch.sigmoid(all_y)).type(torch.int),
        torch.round(torch.sigmoid(all_outputs)).type(torch.int),
    )
    print(
        "[%d] F1: %.3f, T-Loss: %.3f, V-Loss: %.3f"
        % (epoch + 1, f_1, train_loss, np.mean(losses_batches))
    )

    logging.info(
        "[%d] F1: %.3f, T-Loss: %.3f, V-Loss: %.3f"
        % (epoch + 1, f_1, train_loss, np.mean(losses_batches))
    )

    steps_of_val.append(step)
    all_vloss.append(np.mean(losses_batches))
    all_f1.append(f_1)
    all_lr.append(optimizer.param_groups[0]["lr"])

    ax: array[plt.Axes]
    figs, ax = plt.subplots(nrows=2)
    ax[0].plot(
        all_steps,
        all_tloss,
        "-gD",
        label="T-Loss" if epoch == 0 else "",
        markevery=[model_best],
    )
    ax[0].plot(
        steps_of_val,
        all_vloss,
        "-bD",
        label="V-Loss" if epoch == 0 else "",
        markevery=[model_best],
    )
    ax2: plt.Axes = plt.twinx(ax[0])
    ax2.plot(
        steps_of_val,
        all_f1,
        "-rD",
        label="F1" if epoch == 0 else "",
        markevery=[model_best],
    )
    ax2.set_ylim((0, 1.0))
    ax2.grid(True)
    ax[1].plot(steps_of_val, all_lr, color="b", label="LR")
    plt.title("Best model: %d" % model_best)

    plt.savefig(
        os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD, "info.png")
    )

    model.train()

    return np.mean(losses_batches), all_y.shape[0] / np.sum(corrects), f_1


if __name__ == "__main__":

    cma = CommandLineArguments()
    cma.parser.add_argument(
        "-m",
        "--model",
        default="",
        help="Model used for inference",
        required=True,
        type=str,
    )
    cma.parse_args()
    model_name = ModelChoices(cma.args.model)

    config.CHECKPOINT_PATH = os.path.join(
        os.environ["cs_checkpoint_path"], model_name.value
    )
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    config.CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_PATH, config.TASK)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(
        os.path.join(config.CHECKPOINT_PATH, "fold_%d" % config.FOLD), exist_ok=True
    )

    num_classes = 1

    logging.basicConfig(
        filename=os.path.join(
            config.CHECKPOINT_PATH, "fold_%d" % config.FOLD, "info.log"
        ),
        encoding="utf-8",
        level=logging.DEBUG,
    )

    model = get_model(model_name.value)

    model = model.cuda()
    # print(model)

    train_samples, val_samples = get_split()

    dl_train = Mprage2space(
        train_samples,
        config.BATCH_SIZE,
        config.PATCH_SIZE,
        config.NUM_WORKERS,
        seed_for_shuffle=config.FOLD,
        return_incomplete=False,
        shuffle=True,
    )

    transform = get_train_transform(config.PATCH_SIZE)

    mt_train = MultiThreadedAugmenter(
        data_loader=dl_train,
        transform=transform,
        num_processes=config.NUM_WORKERS,
        num_cached_per_queue=4,
        pin_memory=True,
    )

    dl_val = Mprage2space(
        val_samples,
        config.BATCH_SIZE,
        config.PATCH_SIZE,
        config.NUM_WORKERS,
        return_incomplete=False,
        shuffle=False,
    )

    mt_val = MultiThreadedAugmenter(
        data_loader=dl_val,
        transform=Compose([ZeroMeanUnitVarianceTransform()]),
        num_processes=config.NUM_WORKERS,
        num_cached_per_queue=4,
        pin_memory=True,
    )

    # print(f"Amount of validation samples: \t{len(dl_val)}")
    # TODo:
    #   + -1. Add Black as formatter (`Blackd` integration for PyCharm) https://github.com/psf/black https://black.readthedocs.io/en/stable/integrations/editors.html
    #   + 0. Increase epochs --> 400 to 1000 or 1600 for 24h jobs or 18h ?
    #   + 1. Log Steps instead of epochs
    #   + 2. Fix Step size per epoch
    #   + 3. Remove Pretraining for all architectures
    #   + 4. Train with pretraining for Monai & Video ResNet
    #       a. If pretraining is the culprit --> Test other architectures with the longer schedules as well!
    #   5. Test stronger augmentations (side-quest can take a lot of time)
    #       a. Increase Noise
    #       b. MixUp -- Input data interpolation + label interpolation https://arxiv.org/abs/1710.09412
    #           (probably somehwat boring for binary) --> Maybe not really helpful
    #       c. Elastic transforms form Fabis Library
    #       d. Bias-Fields different Augmentations (Maybe try Fabis DA5 data augmentation scheme from Phabricator "development" branch I think --> else ask Fabi?)
    #   6. Add Dropout to the final FC layers? (optional)
    #   7. Test-time augmentation for the final accuracy prediction? (optional)
    #       --> Rotations and mirroring and then eval and see if its better
    #   Track Summarize the results in a Google Drive document so people can comment (Do NOT dump into slack!)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=2e-4, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - config.SCHEDULER_ENTRY, eta_min=1e-10
    )
    scaler = torch.cuda.amp.GradScaler()

    if config.TENSOR_BOARD:
        writer = SummaryWriter(
            os.path.join(
                config.CHECKPOINT_PATH,
                "fold_%d" % config.FOLD,
                "runs/classification/track_stats",
            )
        )

    model.train()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    best_f1 = 0

    for epoch in range(0, config.NUM_EPOCHS):

        train_loss = train_fn(model, criterion, mt_train, optimizer, scheduler, epoch)

        val_loss, corrects, f1 = evaluate(model, epoch, mt_val, train_loss)

        if config.TENSOR_BOARD:
            writer.add_scalar("train_loss", train_loss, global_step=epoch)
            writer.add_scalar("val_loss", val_loss, global_step=epoch)
            writer.add_scalar("val_acc", corrects, global_step=epoch)
            writer.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], global_step=epoch
            )

        if f1 > best_f1:
            best_f1 = f1
            model_best = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model,
                optimizer,
                os.path.join(
                    config.CHECKPOINT_PATH,
                    "fold_%d" % config.FOLD,
                    "model_best.pth.tar",
                ),
            )

    print("Finished Training")
    save_checkpoint(
        model,
        optimizer,
        os.path.join(
            config.CHECKPOINT_PATH, "fold_%d" % config.FOLD, "model_end.pth.tar"
        ),
    )
    model.load_state_dict(best_model_wts)
