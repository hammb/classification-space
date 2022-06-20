#!/usr/bin/env python3

import os
import logging
import csv
import argparse
import numpy as np
import torchvision.models as models
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

toWrite = [
    ["architecture", "IMAGE", "MODEL", "FOLD", "num correct", "num wrong", "TP", "FP", "FN", "TN", "precision", "recall", "f1",
     "sum"]
]

file = open('results.csv', 'w')


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluate(model, mt_test, fold, model_name):
    outputs_folds = {}
    all_outputs = torch.Tensor()
    all_y = torch.Tensor()
    corrects = []
    fold_name = "fold_%d" % fold  # config.FOLD

    load_checkpoint(
        os.path.join(config.CHECKPOINT_PATH, fold_name, "model_best.pth.tar"), model,
        optimizer, lr=2e-4)

    model.eval()

    for batch_idx, batch in enumerate(mt_test):
        x = torch.from_numpy(batch["data"]).to(config.DEVICE)
        y = torch.from_numpy(batch["class"]).to(config.DEVICE)

        all_y = torch.cat([all_y, y.detach().to('cpu')])

        with torch.no_grad():
            outputs = model(x)

            all_outputs = torch.cat([all_outputs, outputs.detach().to('cpu')])

    outputs_folds[fold] = torch.round(torch.sigmoid(all_outputs))

    outputs_ensamble = []
    logging.info(outputs_folds)

    for value_idx in range(len(outputs_folds[list(outputs_folds.keys())[0]])):
        values_in_folds = torch.Tensor()
        for fold in outputs_folds:
            values_in_folds = torch.cat([values_in_folds, outputs_folds[fold][value_idx]])

        ensamble_output = torch.round(torch.mean(values_in_folds)).detach().to('cpu').numpy().min()
        outputs_ensamble.append(ensamble_output)

    outputs_ensamble = np.array(outputs_ensamble)
    all_y_np = []
    for value in all_y:
        all_y_np.append(value.detach().to('cpu').numpy().min())
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

    print('TP: %d, FP: %d, FN: %d, TN: %d' % (tp, fp, fn, tn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('precision: %f, recall: %f' % (precision, recall))

    f1 = 2 * (precision * recall) / (precision + recall)

    print('f1: %f' % f1)

    # corrects.append(torch.sum(torch.round(torch.sigmoid(outputs)) == y).detach().to('cpu').numpy().min())
    # return all_y.shape[0] / np.sum(corrects)

    toWrite.append([model_name, config.TASK, config.TASK, fold, num_correct, num_wrong, tp, fp, fn, tn, precision, recall, f1,
                    num_correct + num_wrong])


if __name__ == '__main__':

    for task in ["mprage", "space", "mprage_3in"]:
        for model_name in ["training_monai_densenet", "training_monai_effnet", "training_monai_resnet", "training_video_resnet"]:
            for fold in [0, 1, 2, 3, 4]:

                config.NUM_INPUT_CHANNELS = 3 if task == "mprage_3in" else 1

                config.TASK = task
                config.FOLD = fold
                config.TRAIN_DIR = input_path = "/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0/" + task + "/test/all_samples"
                config.CHECKPOINT_PATH = os.path.join(os.environ['cs_checkpoint_path'], model_name)
                config.CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_PATH, config.TASK)
                num_classes = 1

                logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

                if model_name == "training_monai_densenet":
                    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
                    model.features[0] = nn.Conv3d(config.NUM_INPUT_CHANNELS, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                                  padding=(config.NUM_INPUT_CHANNELS, 3, 3), bias=False)
                elif model_name == "training_monai_effnet":
                    model = EfficientNetBN("efficientnet-b0", pretrained=False, spatial_dims=3, in_channels=1, num_classes=1)
                    model._conv_stem = nn.Conv3d(config.NUM_INPUT_CHANNELS, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                                 padding=(config.NUM_INPUT_CHANNELS, 3, 3), bias=False)
                    model._fc = nn.Linear(1280, num_classes, bias=True)
                elif model_name == "training_monai_resnet":
                    model = resnet18(
                        pretrained=False,
                        spatial_dims=3,
                    )
                    model.conv1 = nn.Conv3d(config.NUM_INPUT_CHANNELS, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                            padding=(config.NUM_INPUT_CHANNELS, 3, 3), bias=False)
                    model.fc = nn.Linear(512, num_classes, bias=True)
                else:
                    model = models.video.r3d_18(pretrained=True)
                    model.fc = nn.Linear(512, num_classes)
                    model.stem[0] = nn.Conv3d(config.NUM_INPUT_CHANNELS, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                              padding=(config.NUM_INPUT_CHANNELS, 3, 3), bias=False)

                # print(model)

                optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

                model = model.cuda()

                test_samples = os.listdir(input_path)

                dl_test = Mprage2space(test_samples, config.BATCH_SIZE, config.PATCH_SIZE, config.NUM_WORKERS,
                                       return_incomplete=False, shuffle=False)

                mt_test = MultiThreadedAugmenter(
                    data_loader=dl_test,
                    transform=Compose([]),
                    num_processes=config.NUM_WORKERS,
                )

                evaluate(model, mt_test, fold, model_name)

    with file:
        writer = csv.writer(file)

        for row in toWrite:
            writer.writerow(row)