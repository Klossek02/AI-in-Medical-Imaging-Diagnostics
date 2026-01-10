from .tri_conv_unext import build_tri_conv_unext
from .unet_plusplus import build_unet_plusplus

models = {
    'tri_conv_unext': build_tri_conv_unext,
    'unet_plusplus_resnet34': lambda device, in_channels, classes, base_c, bilinear, **kwargs: build_unet_plusplus(device, encoder='resnet34', in_channels=in_channels, classes=classes)
}

def build_model(device, model_type='tri_conv_unext', in_channels=1, classes=2, base_c=32, bilinear=True, **kwargs):
    return models[model_type](device, in_channels, classes, base_c, bilinear, **kwargs)

__all__ = ["build_model", "models"]