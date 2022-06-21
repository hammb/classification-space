# Create CNN Model
import numpy as np
import torch
from torch import nn
import classification_config as config


def _conv_layer_set(in_c, out_c):
    conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
    )
    return conv_layer


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv_layer1 = _conv_layer_set(1, 32)
        self.conv_layer2 = _conv_layer_set(32, 64)
        self.fc1 = nn.Linear(746496, 1)
        self.fc2 = nn.Linear(1, num_classes)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out


inputs = torch.randn((1, 1, 16, 224, 224)).cuda()
labels = torch.ones(1).cuda()

model = CNNModel(1)
model = model.cuda()

error = nn.BCEWithLogitsLoss()

inputs1 = torch.Tensor(np.load(
    "/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0/space/train/all_samples/m_2PK9Y85K_1.npy",
    mmap_mode="r") / config.MAX_VALUE)[None, None, :, :, :].cuda()
labels1 = torch.ones(1).cuda()
inputs2 = torch.Tensor(np.load(
    "/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0/space/train/all_samples/nm_MPHH2HXV_80.npy",
    mmap_mode="r") / config.MAX_VALUE)[None, None, :, :, :].cuda()
labels2 = torch.zeros(1).cuda()
# test forward
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

model.train()
for epoch in range(1000):
    with torch.cuda.amp.autocast():
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs1)
        loss = criterion(outputs, labels1[None, :])
        loss.backward()
        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs2)
        loss = criterion(outputs, labels2[None, :])
        loss.backward()
        optimizer.step()

        # print statistics
        print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')
