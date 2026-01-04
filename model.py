import torch
import segmentation_models_pytorch as smp

def get_model(device, encoder='resnet34', in_channels=1, classes=2):
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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    try:
        model = get_model(device)
        
        test_input = torch.randn(2, 1, 256, 256).to(device)  # input: [Batch, Channel, Height, Width]
        
        output = model(test_input)
        
        print(f"Model works correctly.")
        print(f" Input shape: {test_input.shape}")
        print(f" Output shape: {output.shape}") # exp: [2, 2, 256, 256])
        
    except Exception as e:
        print(f"Model error: {e}")