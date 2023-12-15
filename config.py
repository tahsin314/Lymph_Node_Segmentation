from models.mit.mit_PLD_b4 import mit_PLD_b4
from models.unet import U_Net
from models.mit.mit_PLD_b2 import mit_PLD_b2
from models.CaraNet.caranet import caranet
from models.FCBFormer.FCBFormer import FCBFormer
from models.DUCKNet.DUCKNet import DuckNet
import cv2
import albumentations as A


''' You only need to change the config_parmas dictionary'''
config_params = dict(
    data_dir = "../DATA/lymph_node/ct_221_npz", #"../DATA/Clouds/38-Cloud_training"
    model_dir = 'model_dir',
    model_name = 'DUCKNet',
    n_fold = 5,
    fold = 1,
    sz = 512,
    num_channels = 1,
    threshold = 0.25,
    dataset = 'LN Segmentation',
    lr = 1.5e-4,
    eps = 1e-5,
    weight_decay = 1e-5,
    n_epochs = 60,
    bs = 24,
    gradient_accumulation_steps = 2,
    SEED = 2023,
    sampling_mode = None, #upsampling
    pretrained = False,
    mixed_precision = False
    )

model_params = dict(

    UNet = U_Net(config_params['num_channels'], 1),
    ssformer_S = mit_PLD_b2(class_num=1, in_chans=config_params['num_channels']),
    ssformer_L = mit_PLD_b4(class_num=1),
    CaraNet = caranet(in_chans=config_params['num_channels']),
    FCBFormer = FCBFormer(size=config_params['sz']),
    DUCKNet = DuckNet(in_chans=config_params['num_channels'], starting_filters=10)
    )

aug_config = dict(

    train_aug = A.Compose([
  A.ShiftScaleRotate(p=0.9,rotate_limit=30, border_mode= cv2.BORDER_CONSTANT, value=[0, 0, 0], scale_limit=0.25),
    A.OneOf([
    A.Cutout(p=0.3, max_h_size=int(config_params['sz'])//16, max_w_size=int(config_params['sz'])//16, num_holes=10, fill_value=0),
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

color_config = dict(
  RED = '\033[91m',
  GREEN = '\033[92m',
  YELLOW = '\033[93m',
  BLUE = '\033[94m',
  MAGENTA = '\033[95m',
  CYAN = '\033[96m',
  RESET = '\033[0m',
  BLACK = '\033[30m',
  WHITE = '\033[97m',
  ORANGE = '\033[38;5;208m',  # Orange color
  PURPLE = '\033[35m',
  GREEN_BG = '\033[42m',  # Green background
  BOLD = '\033[1m',  # Bold text
  ITALIC = '\033[3m',  # Italic text
  UNDERLINE = '\033[4m'  # Underlined text 

)