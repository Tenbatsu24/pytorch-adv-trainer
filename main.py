import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import os

from trainer import train_suit, robust_test_suit


def one_to_three(__ims):
    return F.conv2d(__ims, torch.ones((3, 1, 1, 1)).to(__ims.device))


batch_size = 128

load_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2859,), (0.3530,)),
])

training_set = FashionMNIST(os.path.join('.', 'datas'), train=True, download=True, transform=load_transform)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

testing_set = FashionMNIST(os.path.join('.', 'datas'), train=False, download=True, transform=load_transform)
test_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

net = torchvision.models.efficientnet_b0(pretrained=False, num_classes=10)
optim = torch.optim.Adam(net.parameters(), 0.03)
crit = nn.CrossEntropyLoss()

model_names = ['eff0_fm_none', 'eff0_adv_fm_none']

for model_name in model_names:
    path_for_trained_model = os.path.join('.', 'models', f'{model_name}.pth')

    train = False
    if train:
        train_suit(model=net, num_epochs=10, training_data_loader=train_loader, data_augmentation=one_to_three,
                   optimiser=optim, criterion=crit, use_cuda=True, save_model=True, path_to_file=path_for_trained_model,
                   adv_training=False,
                   testing_data_loader=test_loader, testing_length=len(testing_set), test_aug=one_to_three
                   )

    test_robust = True
    if test_robust:
        robust_test_suit(model=net, path_to_file=path_for_trained_model,
                         testing_data_loader=test_loader, testing_length=len(testing_set), test_aug=one_to_three
                         )
