import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Model import *
from Data import *
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("Configs/base.yaml", "r") as file:
    config = yaml.safe_load(file)

IMG_TRAIN_DIR = config["path"]["img_train"]
HM_TRAIN_DIR  = config["path"]["hm_train"]
IMG_VAL_DIR   = config["path"]["img_val"]
HM_VAL_DIR    = config["path"]["hm_val"]
IMG_TEST_DIR  = config["path"]["img_test"]
HM_TEST_DIR   = config["path"]["hm_test"]

BATCH_SIZE = config["training"]["batch_size"]
PATIENCE_LR = config["training"]["patience_lr"]
PATIENCE_ES = config["training"]["patience_es"]
LOSS_WEIGHTS = config["loss"]["loss_weights"]

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

optimizer_name = config["training"]["optimizer"]["name"]
optimizer_params = config["training"]["optimizer"]["params"]
OptimizerClass = getattr(optim, optimizer_name)
optimizer = OptimizerClass(model.parameters(), **optimizer_params)

criterion = nn.CrossEntropyLoss()

