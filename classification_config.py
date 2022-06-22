#!/usr/bin/env python3
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FOLD = 0
TASK = ""
CHECKPOINT_PATH = ""
TRAIN_DIR = ""

NUM_WORKERS = 24
NUM_EPOCHS = 500
SCHEDULER_ENTRY = 200
PATCH_SIZE = (16, 224, 224)
BATCH_SIZE = 2
# TRAIN_DIR = os.path.join("/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0", TASK,
#                          "train/all_samples")
# TEST_DIR = os.path.join("/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0", TASK,
#                         "/test/all_samples")

MAX_VALUE = 1582 if TASK == "space" else 987

CHECKPOINT = "model.pth.tar"
CHECKPOINT_BEST = "model_best.pth.tar"

NUM_INPUT_CHANNELS = 1
FOLD = 0

TENSOR_BOARD = False