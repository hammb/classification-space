import os
import random

import numpy as np

import classification_config as config
from batchgenerators.transforms.abstract_transforms import Compose
from efficientnet_pytorch_3d import EfficientNet3D
from dataloading.claspace_dataloader import ClaSpace
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_train_transform(patch_size):
    tr_transforms = []
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

    for batch_idx, batch in enumerate(mt_train):
        x = batch[0][:, None, :, :, :].to(config.DEVICE)
        y = batch[1].to(config.DEVICE)

        with torch.cuda.amp.autocast():
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            loss = criterion(outputs.type(torch.float32)[:, :, None], y.type(torch.float32))
            loss.backward()
            optimizer.step()


def evaluate(model, epoch, fold, mt_val):
    model.eval()

    mean_acc = []
    for batch_idx, batch in enumerate(mt_val):
        x = batch[0][:, None, :, :, :].to(config.DEVICE)
        y = batch[1].to(config.DEVICE)
        with torch.no_grad():
            outputs = model(x)
            mean_acc.append(
                100 / y.size()[0] * torch.sum(torch.round(outputs)[:, :, None] == y).detach().to('cpu').numpy().min())
    print('[%d] Acc.: %.3f' % (epoch + 1, np.mean(np.array(mean_acc))))
    model.train()


if __name__ == '__main__':

    model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 1}, in_channels=1)

    model = model.cuda()
    train_samples, val_samples = get_split()

    train_dataset = ClaSpace(root_dir=config.TRAIN_DIR, list_samples=train_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_dataset = ClaSpace(root_dir=config.TRAIN_DIR, list_samples=val_samples)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()
    model.train()

    for epoch in range(0, config.NUM_EPOCHS):
        train_fn(
            model, criterion, val_loader, optimizer, epoch
        )

        evaluate(model, epoch, config.FOLD, val_loader)

    print('Finished Training')
