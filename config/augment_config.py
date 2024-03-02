from config.config import config_params
import cv2
import albumentations as A

aug_config = dict(

    train_aug = A.Compose([
  A.ShiftScaleRotate(p=0.9,rotate_limit=30, border_mode= cv2.BORDER_CONSTANT, value=[0, 0, 0], scale_limit=0.25),
    A.OneOf([
    # A.Cutout(p=0.3, max_h_size=int(config_params['sz'])//16, max_w_size=int(config_params['sz'])//16, num_holes=10, fill_value=0),
    # GridMask(num_grid=7, p=0.7, fill_value=0)
    ], p=0.20),
    A.RandomSizedCrop(min_max_height=(int(int(config_params['sz'])*0.7), int(int(config_params['sz'])*0.8)), height=int(config_params['sz']), width=int(config_params['sz']), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
    # RandomAugMix(severity=3, width=1, alpha=1., p=0.3), 
    # OneOf([
    #     # Equalize(p=0.2),
    #     Posterize(num_bits
    #     =4, p=0.4),
    #     Downscale(0.40, 0.80, cv2.INTER_LINEAR, p=0.3)                  
    #     ], p=0.2),
    A.OneOf([
        A.GaussNoise(var_limit=0.1),
        A.Blur(),
        A.GaussianBlur(blur_limit=3),
        # RandomGamma(p=0.7),
        ], p=0.1),
    A.HueSaturationValue(p=0.4),
    A.HorizontalFlip(0.4),
    # Normalize(always_apply=True)
    ]
      ),
val_aug = A.Compose([A.Normalize(always_apply=True)])
)

