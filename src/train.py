import torch
import wandb
import pandas as pd
from tqdm import tqdm
import os
from src.config import Config
from src.model import get_model
from src.loss import get_loss_fn
from src.metrics import calculate_metrics
from src.plotting import plot_training_results, visualize_predictions
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from torch.utils.data import DataLoader
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

def train_one_epoch(model, loader, optimizer, loss_fn, device, config: Config):
    """
    Trains the model for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to run on
        config: Config object
    
    Returns:
        Average loss, dice score, accuracy, and IoU for the epoch
    """
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_acc = 0.0
    epoch_iou = 0.0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, subbatch in enumerate(loop):
        images = subbatch["image"]
        masks = subbatch["label"]
        images = images.to(device)
        masks = masks.to(device)
        
        # MONAI Loss requires channel dimension in mask: [B, 1, H, W]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(outputs, dim=1)
        dice, acc, iou = calculate_metrics(config, preds, masks)
        
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        epoch_acc += acc.item()
        epoch_iou += iou.item()

        loop.set_postfix(loss=loss.item(), dice=dice.item(), iou=iou.item())
    
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_acc / len(loader), epoch_iou / len(loader)

def validate(model, loader, loss_fn, device, config: Config, vis_num=3, model_output_callback: callable = None):
    """
    Validates the model on validation data and visualizes few examples of predictions.
    
    Args:
        model: Model to validate
        loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to run on
        vis_num: Number of examples to visualize
    
    Returns:
        Average loss, dice score, accuracy, and IoU for the validation set
        Visualized predictions
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_acc = 0.0
    val_iou = 0.0
    visualized_predictions = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            masks = batch["label"]
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            outputs = sliding_window_inference(inputs=images, predictor=model, roi_size=config.dataloader.crop_size, sw_batch_size=config.dataloader.batch_size, overlap=config.dataloader.sw_overlap, mode="gaussian")
            preds = torch.argmax(outputs, dim=1)

            if model_output_callback is not None:
                outputs = model_output_callback(outputs)
                og_outputs = outputs.clone()
                og_preds = torch.argmax(og_outputs, dim=1)
            loss = loss_fn(outputs, masks)
            
            dice, acc, iou = calculate_metrics(config, preds, masks)
            
            val_loss += loss.item()
            val_dice += dice.item()
            val_acc += acc.item()
            val_iou += iou.item()
            # Visualize prediction
            for i in range(images.shape[0]):
                if len(visualized_predictions) < vis_num:
                    visualized_predictions.append((images[i, 0].cpu().numpy(), masks[i].cpu().numpy().squeeze(), preds[i].cpu().numpy().squeeze()))
                    if model_output_callback is not None:
                        visualized_predictions.append((images[i, 0].cpu().numpy(), masks[i].cpu().numpy().squeeze(), og_preds[i].cpu().numpy().squeeze()))
    
    return val_loss / len(loader), val_dice / len(loader), val_acc / len(loader), val_iou / len(loader), visualized_predictions

def test(config: Config, test_loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device, vis_num: int = 3, save_dir: str = "models", use_wandb: bool = False, model_output_callback: callable = None):
    """
    Tests the model on test data.
    
    Args:
        config: Config object
        test_loader: DataLoader for test data
        loss_fn: Loss function
        device: Device to run on
        vis_num: Number of examples to visualize
        save_dir: Directory to save the model
    """
    print(f"🔥 Testing model on {config.training.device}")
    print(f"🏗️  Architecture: {config.model.model_type} | Loss: {config.training.loss_fn_type}")

    # Load model
    model = get_model(config, device=config.training.device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=config.training.device))
    model.eval()

    # Validate
    test_pred_path = os.path.join(save_dir, "test_predictions")
    test_loss, test_dice, test_acc, test_iou, visualized_predictions = validate(model, test_loader, loss_fn, device, config, vis_num, model_output_callback)
    image_paths = visualize_predictions(visualized_predictions, test_pred_path)
    if use_wandb:
        wandb.log({
            "final_predictions": [wandb.Image(path) for path in image_paths]
        })
        wandb.log({
            'test/loss': test_loss,
            'test/dice': test_dice,
            'test/accuracy': test_acc,
            'test/iou': test_iou,
        })
    # Save test metrics to csv
    test_metrics_df = pd.DataFrame([{
        'run_identifier': config.run_identifier,
        'test/loss': test_loss,
        'test/dice': test_dice,
        'test/accuracy': test_acc,
        'test/iou': test_iou,
    }])
    test_metrics_df.to_csv(os.path.join(save_dir, "test_metrics.csv"), index=False)
    print(f"🏁 Test finished!")
    print(f"🎯 Dice: {test_dice:.4f}")
    print(f"✅ Acc: {test_acc:.4f}")
    print(f"💡 IoU: {test_iou:.4f}")
    print(f"💾 Saved test predictions to {test_pred_path}")

def train(
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    save_dir: str = "models",
    use_wandb: bool = False
):
    """
    Main training function following the pattern from preprocessing.py and dataloader.py.
    
    Args:
        config: Config object containing all configuration
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: Optional DataLoader for test data (not used during training)
        save_dir: Directory to save the model
        use_wandb: If True, log metrics to wandb (default: False)
    """
    print(f"🔥 Launching training on {config.training.device}")
    print(f"🏗️  Architecture: {config.model.model_type} | Loss: {config.training.loss_fn_type}")
    
    # Set random seed for reproducibility
    if config.training.seed is not None:
        set_determinism(seed=config.training.seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Build model
    device = torch.device(config.training.device)
    model = get_model(config, device=device)

    if use_wandb:
        wandb.log({
            'number_of_parameters': sum(p.numel() for p in model.parameters()),
        })
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create loss function
    loss_fn = get_loss_fn(config.training.loss_fn_type, **config.training.loss_fn_params)
    
    # Training loop
    best_val_dice = 0.0
    early_stopping_counter = 0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    train_accs, val_accs = [], []
    train_ious, val_ious = [], []
    print(f"\n🚀 Starting training for {config.training.num_epochs} epochs...")
    
    for epoch in range(config.training.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.training.num_epochs} ---")
        
        # Train
        t_loss, t_dice, t_acc, t_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, config
        )
        
        # Validate
        v_loss, v_dice, v_acc, v_iou, visualized_predictions = validate(
            model, val_loader, loss_fn, device, config, vis_num=config.training.vis_num
        )

        # Visualize predictions
        image_paths = visualize_predictions(visualized_predictions, os.path.join(save_dir, f"epoch_{epoch+1}_predictions"))
        if use_wandb:
            wandb.log({
                "predictions": [wandb.Image(path) for path in image_paths]
            }, step=epoch+1)

        # Update learning rate scheduler
        if config.training.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(v_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_dices.append(t_dice)
        val_dices.append(v_dice)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        train_ious.append(t_iou)
        val_ious.append(v_iou)
        # Logging
        print(f"📉 Loss -> Train: {t_loss:.4f} | Val: {v_loss:.4f}")
        print(f"🎯 Dice -> Train: {t_dice:.4f} | Val: {v_dice:.4f}")
        print(f"✅ Acc  -> Train: {t_acc:.4f} | Val: {v_acc:.4f}")
        print(f"💡 IoU -> Train: {t_iou:.4f} | Val: {v_iou:.4f}")
        print(f"⚡ LR: {current_lr:.2e}")
        
        # WandB logging
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': t_loss,
                'train/dice': t_dice,
                'train/accuracy': t_acc,
                'train/iou': t_iou,
                'val/loss': v_loss,
                'val/dice': v_dice,
                'val/accuracy': v_acc,
                'val/iou': v_iou,
                'learning_rate': current_lr,
            }, step=epoch+1)

        # Append metrics to csv
        metrics_df = pd.DataFrame([{
            'epoch': epoch + 1,
            'train/loss': t_loss,
            'train/dice': t_dice,
            'train/accuracy': t_acc,
            'train/iou': t_iou,
            'val/loss': v_loss,
            'val/dice': v_dice,
            'val/accuracy': v_acc,
            'val/iou': v_iou,
            'learning_rate': current_lr,
        }])
        metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), mode='a', header=not os.path.exists(os.path.join(save_dir, "metrics.csv")), index=False)
        
        # Save best model
        if v_dice > best_val_dice + config.training.early_stopping_delta:
            diff = v_dice - best_val_dice
            best_val_dice = v_dice
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth")
            )
            print(f"💾 Saved new best model! (Dice improved by: {diff:.4f})")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.training.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training results
    plot_path = os.path.join(save_dir, "training_results.png")
    plot_training_results(train_losses, val_losses, train_dices, val_dices, plot_path)

    print(f"\n✅ Training finished!")
    print(f"Best Dice: {best_val_dice:.4f}")
    
    # Test
    test(config, test_loader, loss_fn, device, vis_num=config.training.vis_num, save_dir=save_dir, use_wandb=use_wandb)

    print(f"\n✅ All done!")
    
