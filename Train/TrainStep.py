import torch
from ignite.handlers import ReduceLROnPlateauScheduler
from sympy.simplify.hyperexpand import ReduceOrder
from torch import nn, optim
from torch.utils.data import DataLoader
from Model import *
from Data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_TRAIN_DIR = '/kaggle/input/images-heatmaps-separate/npy_images/train'
HM_TRAIN_DIR  = '/kaggle/input/images-heatmaps-separate/npy_heatmaps/train'
IMG_VAL_DIR   = '/kaggle/input/images-heatmaps-separate/npy_images/val'
HM_VAL_DIR    = '/kaggle/input/images-heatmaps-separate/npy_heatmaps/val'
IMG_TEST_DIR  = '/kaggle/input/images-heatmaps-separate/npy_images/test'
HM_TEST_DIR   = '/kaggle/input/images-heatmaps-separate/npy_heatmaps/test'

LR = 1e-4
BATCH_SIZE = 2
EPOCHS = 300
PATIENCE_LR = 10
PATIENCE_ES = 30
LOSS_WEIGHTS = [.1, .5, 1.]

train_dataset = BSSPDataset(image_path=IMG_TRAIN_DIR,
                            heatmap_path=HM_TRAIN_DIR,
                            augmentation=BSSPAugmentation())
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

validation_dataset = BSSPDataset(image_path=IMG_VAL_DIR,
                                 heatmap_path=HM_VAL_DIR)
validation_loader = DataLoader(dataset=validation_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=4,
                          drop_last=True)

test_dataset = BSSPDataset(image_path=IMG_TEST_DIR,
                            heatmap_path=HM_TEST_DIR)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=4)

model = BSSPNet(in_channels=3,
                base=16,
                scale=2,
                num_classes=20).to(device=device)
optimizer = optim.RMSprop(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()