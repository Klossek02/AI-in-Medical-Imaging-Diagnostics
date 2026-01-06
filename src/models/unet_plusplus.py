import torch
import segmentation_models_pytorch as smp

def build_unet_plusplus(device, encoder='resnet34', in_channels=1, classes=2):
    """
    The function builds and returns a segmentation model - Unet++ with scSE attention.
    Encoder: ResNet34.
    """
    
    print(f"Building model: Unet++ with encoder: {encoder} and attention: scSE")
    
    model = smp.UnetPlusPlus(
        encoder_name=encoder, 
        encoder_weights="imagenet", 
        in_channels=in_channels, 
        classes=classes,
        decoder_attention_type='scse' 
    )

    return model.to(device)