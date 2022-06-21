import numpy as np
import torch
import torchvision.models as models
from torch import nn
import classification_config as config
num_classes = 1

model = models.video.r3d_18(pretrained=True)
model.fc = nn.Linear(512, num_classes)
model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
model = model.cuda()

print(model)

inputs1 = torch.Tensor(np.load("/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0/space/train/all_samples/m_2PK9Y85K_1.npy", mmap_mode="r") / config.MAX_VALUE)[None, None, :, :, :].cuda()
labels1 = torch.ones(1).cuda()

#inputs1 = torch.stack((inputs1, inputs1, inputs1), dim=1)

inputs2 = torch.Tensor(np.load("/home/AD/b556m/data/classification_space/classification_space_preprocessed_b0/space/train/all_samples/nm_MPHH2HXV_80.npy", mmap_mode="r") / config.MAX_VALUE)[None, None, :, :, :].cuda()
labels2 = torch.zeros(1).cuda()

#inputs2 = torch.stack((inputs2, inputs2, inputs2), dim=1)

# test forward
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)

model.train()
for epoch in range(2):
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
