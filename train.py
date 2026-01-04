import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

# Importujemy Twoje moduły
from dataloader import get_dataloaders
from model import get_model 

# Używamy zwycięskiej funkcji straty
from monai.losses import DiceFocalLoss
from monai.utils import set_determinism

# --- KONFIGURACJA ---
DATA_DIR = "data/preprocessed_v2" # lub "data/v2/preprocessed_revised" - sprawdź gdzie masz dane
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unet++ jest duży, więc Batch Size 8 jest bezpieczniejszy dla pamięci GPU
BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
NUM_EPOCHS = 15        

set_determinism(seed=33)

# --- FUNKCJE POMOCNICZE ---
def calculate_metrics(preds, targets):
    """Oblicza Dice Score i Accuracy dla klasy 1 (kręgosłup)"""
    smooth = 1e-6
    # Spłaszczamy tensory
    preds_flat = preds.view(-1).float()
    targets_flat = targets.view(-1).float()
    
    # Dice
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    # Accuracy
    correct = (preds_flat == targets_flat).sum()
    acc = correct / len(targets_flat)
    
    return dice, acc

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_acc = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)
        
        # MONAI Loss wymaga wymiaru kanału w masce: [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        
        # 1. Forward
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # 2. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 3. Metryki (dla człowieka)
        # Zamieniamy logity na konkretne klasy (0 lub 1)
        preds = torch.argmax(outputs, dim=1) 
        
        # Do metryk potrzebujemy maski bez wymiaru kanału [B, H, W] lub spłaszczonej
        dice, acc = calculate_metrics(preds, masks)
        
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        epoch_acc += acc.item()
        
        loop.set_postfix(loss=loss.item(), dice=dice.item())
        
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_acc / len(loader)

def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    val_acc = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            preds = torch.argmax(outputs, dim=1)
            dice, acc = calculate_metrics(preds, masks)
            
            val_loss += loss.item()
            val_dice += dice.item()
            val_acc += acc.item()
            
    return val_loss / len(loader), val_dice / len(loader), val_acc / len(loader)

def plot_training_results(train_losses, val_losses, train_dices, val_dices):
    """Rysuje wykresy Loss i Dice"""
    plt.figure(figsize=(12, 5))
    
    # Wykres Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss (Im mniej tym lepiej)")
    plt.legend()
    plt.grid(True)
    
    # Wykres Dice
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label="Train Dice")
    plt.plot(val_dices, label="Val Dice")
    plt.title("Dice Score (Im więcej tym lepiej)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(SAVE_DIR, "training_results.png"))
    plt.close()

def main():
    print(f"🔥 Launching training on {DEVICE}")
    print(f"🏗️  Architecture: Unet++ (ResNet34) | Loss: DiceFocalLoss")
    
    # 1. Wczytanie danych (korzystamy z Twojego nowego dataloader.py)
    print("⏳ Loading dataloaders...")
    try:
        # get_dataloaders zwraca teraz 3 loadery (train, val, test)
        train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"❌ Error loading dataloaders: {e}")
        return

    # 2. Budowa modelu (korzystamy z Twojego nowego model.py)
    # Funkcja get_model ma domyślne parametry ustawione na Unet++ i ResNet34
    model = get_model(device=DEVICE)
    
    # 3. Optymalizator
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # 4. Scheduler (zmniejsza LR jak loss przestaje spadać)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # 5. Funkcja straty (Zwycięska konfiguracja)
    loss_fn = DiceFocalLoss(softmax=True, to_onehot_y=True, lambda_dice=1.0, lambda_focal=2.0)
    
    # --- PĘTLA TRENINGOWA ---
    best_val_dice = 0.0 # Śledzimy Dice, nie Loss!
    
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Trenuj
        t_loss, t_dice, t_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        
        # Waliduj
        v_loss, v_dice, v_acc = validate(model, val_loader, loss_fn, DEVICE)
        
        # Scheduler krok
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Zapisywanie historii
        train_losses.append(t_loss); val_losses.append(v_loss)
        train_dices.append(t_dice);  val_dices.append(v_dice)
        
        # Logowanie
        print(f"📉 Loss -> Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        print(f"🎯 Dice -> Train: {t_dice:.4f} | Val: {v_dice:.4f}")
        print(f"⚡ LR: {current_lr:.2e}")
        
        # Zapisywanie najlepszego modelu (według Dice Score)
        if v_dice > best_val_dice:
            diff = v_dice - best_val_dice
            best_val_dice = v_dice
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"💾 Saved new best model! (Dice improved by: {diff:.4f})")
            
    # Zapisz model końcowy
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model.pth"))
    
    # Rysuj wykresy
    plot_training_results(train_losses, val_losses, train_dices, val_dices)
    
    print("\n✅ Training finished!")
    print(f"Best Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()