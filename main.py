import os

from efficientnet_pytorch_3d import EfficientNet3D
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)

model = model.cuda()
inputs = torch.randn((1, 1, 16, 244, 244)).cuda()
labels = torch.tensor([0]).cuda()
# test forward
num_classes = 1

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(100):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')
