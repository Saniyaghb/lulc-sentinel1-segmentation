"""
models.py - builds the segmentation network (segmentation_models_pytorch) from the architecture and
encoder names. The final model is a U-Net with a ResNet50 encoder; other architectures were compared.
"""
import segmentation_models_pytorch as smp

ARCHS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "segformer": smp.Segformer,
}


def build_model(arch: str, encoder: str, in_channels: int, classes: int, pretrained: bool = False):
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'. Choose from: {', '.join(ARCHS)}")
    return ARCHS[arch](encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                       in_channels=in_channels, classes=classes)