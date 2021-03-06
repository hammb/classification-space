#!/usr/bin/env python3

import os

import argparse
from enum import Enum

import numpy as np
import torchvision.models as models
from batchgenerators.transforms.sample_normalization_transforms import (
    ZeroMeanUnitVarianceTransform,
)
from torch import nn
import classification_config as config
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from dataloading.batchgenerators_mprage2space import Mprage2space
import torch
from command_line_arguments.command_line_arguments import CommandLineArguments
import monai
from monai.networks.nets import EfficientNetBN
from monai.networks.nets import resnet18

from smpl_models import resnet


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


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluate(model, mt_test):
    outputs_folds = {}
    all_outputs = torch.Tensor()
    all_y = torch.Tensor()
    corrects = []
    first_loop = True
    for fold in sorted(os.listdir(config.CHECKPOINT_PATH)):

        if not fold == "fold_3":
            continue

        load_checkpoint(
            os.path.join(config.CHECKPOINT_PATH, fold, "model_best.pth.tar"),
            model,
            optimizer,
            lr=2e-4,
        )

        model.eval()

        for batch_idx, batch in enumerate(mt_test):
            x = torch.from_numpy(batch["data"]).to(config.DEVICE)
            y = torch.from_numpy(batch["class"]).to(config.DEVICE)

            if first_loop:
                all_y = torch.cat([all_y, y.detach().to("cpu")])

            with torch.no_grad():
                outputs = model(x)

                all_outputs = torch.cat([all_outputs, outputs.detach().to("cpu")])

        outputs_folds[fold] = torch.round(torch.sigmoid(all_outputs))
        all_outputs = torch.Tensor()

        if first_loop:
            first_loop = False

    outputs_ensamble = []

    for value_idx in range(len(outputs_folds[list(outputs_folds.keys())[0]])):
        values_in_folds = torch.Tensor()
        for fold in outputs_folds:
            values_in_folds = torch.cat(
                [values_in_folds, outputs_folds[fold][value_idx]]
            )

        ensamble_output = (
            torch.round(torch.mean(values_in_folds)).detach().to("cpu").numpy().min()
        )
        outputs_ensamble.append(ensamble_output)

    outputs_ensamble = np.array(outputs_ensamble)
    all_y_np = []
    for value in all_y:
        all_y_np.append(value.detach().to("cpu").numpy().min())
    all_y_np = np.array(all_y_np)

    # print(all_y_np)
    # print(outputs_ensamble)

    num_correct = np.sum(outputs_ensamble == all_y_np)

    print(num_correct)

    num_wrong = len(outputs_ensamble) - num_correct

    print(num_wrong)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(len(all_y_np)):
        if all_y_np[i] == 1 and outputs_ensamble[i] == 1:
            tp = tp + 1

        if all_y_np[i] == 1 and outputs_ensamble[i] == 0:
            fn = fn + 1

        if all_y_np[i] == 0 and outputs_ensamble[i] == 1:
            fp = fp + 1

        if all_y_np[i] == 0 and outputs_ensamble[i] == 0:
            tn = tn + 1

    print("TP: %d, FP: %d, FN: %d, TN: %d" % (tp, fp, fn, tn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("precision: %f, recall: %f" % (precision, recall))

    f1 = 2 * (precision * recall) / (precision + recall)

    print("f1: %f" % f1)

    # corrects.append(torch.sum(torch.round(torch.sigmoid(outputs)) == y).detach().to('cpu').numpy().min())
    # return all_y.shape[0] / np.sum(corrects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        default=2,
        help="Input path for inference",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default=2,
        help="Output path for results from inference",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-t", "--task", default="", help="Task to train", required=True, type=str
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="Model used for inference",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    model_name = ModelChoices(args.model).value
    config.TASK = args.task
    config.TRAIN_DIR = input_path = args.input_path
    config.CHECKPOINT_PATH = os.path.join(os.environ["cs_checkpoint_path"], args.model)
    config.CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_PATH, config.TASK)
    output_path = args.output_path

    num_classes = 1
    model = get_model(model_name)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=2e-4, momentum=0.9, weight_decay=5e-4
    )

    test_samples = os.listdir(input_path)

    dl_test = Mprage2space(
        test_samples,
        config.BATCH_SIZE,
        config.PATCH_SIZE,
        config.NUM_WORKERS,
        return_incomplete=False,
        shuffle=False,
    )

    mt_test = MultiThreadedAugmenter(
        data_loader=dl_test,
        transform=Compose([ZeroMeanUnitVarianceTransform()]),
        num_processes=config.NUM_WORKERS,
    )

    evaluate(model, mt_test)
