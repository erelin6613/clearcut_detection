import segmentation_models_pytorch as smp
import torch
from models.season_prediction.model import FPN_double_output
from models.autoencoder.model import Autoencoder_Unet


def get_satellite_pretrained_resnet(model_weights_path, encoder_name='resnet50'):
    model = Autoencoder_Unet(encoder_name=encoder_name)
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.encoder


def get_model(model_weights_path, name='fpn50'):
    if name == 'unet34':
        return smp.Unet('resnet34', encoder_weights='imagenet')
    elif name == 'unet50':
        return smp.Unet('resnet50', encoder_weights='imagenet')
    elif name == 'unet101':
        return smp.Unet('resnet101', encoder_weights='imagenet')
    elif name == 'linknet34':
        return smp.Linknet('resnet34', encoder_weights='imagenet')
    elif name == 'linknet50':
        return smp.Linknet('resnet50', encoder_weights='imagenet')
    elif name == 'fpn34':
        return smp.FPN('resnet34', encoder_weights='imagenet')
    elif name == 'fpn50':
        return smp.FPN('resnet50', encoder_weights='imagenet')
    elif name == 'fpn101':
        return smp.FPN('resnet101', encoder_weights='imagenet')
    elif name == 'pspnet34':
        return smp.PSPNet('resnet34', encoder_weights='imagenet', classes=1)
    elif name == 'pspnet50':
        return smp.PSPNet('resnet50', encoder_weights='imagenet', classes=1)
    elif name == 'fpn50_season':
        return FPN_double_output('resnet50', encoder_weights='imagenet')
    elif name == 'fpn50_satellite':
        fpn_resnet50 = smp.FPN('resnet50', encoder_weights=None)
        fpn_resnet50.encoder = get_satellite_pretrained_resnet(model_weights_path)
        return fpn_resnet50
    elif name == 'fpn50_multiclass':
        return smp.FPN('resnet50', encoder_weights='imagenet', classes=3, activation='softmax')
    else:
        raise ValueError("Unknown network")
