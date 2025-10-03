import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model import *
from Data import *
from Test import *
from Train import *
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("Configs/base_config.yaml", "r") as file:
    config = yaml.safe_load(file)

IMG_TRAIN_DIR = config["path"]["img_train"]
HM_TRAIN_DIR  = config["path"]["hm_train"]
IMG_VAL_DIR   = config["path"]["img_val"]
HM_VAL_DIR    = config["path"]["hm_val"]
IMG_TEST_DIR  = config["path"]["img_test"]
HM_TEST_DIR   = config["path"]["hm_test"]

EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
NUM_CLASSES = config["training"]["num_classes"]
PATIENCE_LR = config["training"]["patience_lr"]
PATIENCE_ES = config["training"]["patience_es"]
THRESHOLD = config["training"]["threshold"]
NORMALIZER = config["training"]["normalizer"]
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

scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=PATIENCE_LR, min_lr=1e-6)

# Класичні метрики
classic_metrics = ClassicMetrics(num_classes=NUM_CLASSES, device=device)

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(EPOCHS):
    # --- TRAIN ---
    train_metrics = train_step(model, train_loader, optimizer,
                               criterion, LOSS_WEIGHTS,
                               device, PCK(THRESHOLD), NME(NORMALIZER), classic_metrics,
                               use_amp=True)

    # --- VALIDATION ---
    val_metrics = val_step(model, validation_loader, criterion,
                           device, PCK(THRESHOLD), NME(NORMALIZER), classic_metrics)

    # --- Scheduler ---
    scheduler.step(val_metrics['loss'])  # ReduceLROnPlateau слід викликати після валід. лосс

    # --- Early Stopping ---
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE_ES:
            print(f"Early stopping на {epoch+1}-му епосі")
            break

    # --- Логи ---
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_metrics['loss']:.4f} | PCK: {train_metrics['PCK']:.4f} | NME: {train_metrics['NME']:.4f}")
    print(f"Val   Loss: {val_metrics['loss']:.4f} | PCK: {val_metrics['PCK']:.4f} | NME: {val_metrics['NME']:.4f}")
    print("-"*50)

# --- Тестування після навчання ---
model.load_state_dict(torch.load('best_model.pth'))
test_metrics = test_step(model, test_loader, device, PCK(THRESHOLD), NME(NORMALIZER), classic_metrics)
print("Test metrics:", test_metrics)