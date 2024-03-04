import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from torch.nn.parallel import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from fvcore.nn import FlopCountAnalysis

import wandb
from Datasets.CloudDataset import CloudDataset
from Datasets.DataModule import DataModule

# from config.config import config_params
from config.model_config import model_params
from config.augment_config import aug_config
from config.color_config import color_config
from augmentation import Augmentation
from Datasets.LNDataset import LNDataset
from catalyst.data.sampler import DynamicBalanceClassSampler
from train_module import train_val_class
from utils import save_model, seed_everything
from losses.tversky import tversky_loss, focal_tversky
from losses.dice import dice_loss, dice_lossv2
from losses.focusnetloss import FocusNetLoss
from losses.hybrid import hybrid_loss
from losses.structure_loss import structure_loss, total_structure_loss


parser = argparse.ArgumentParser(description='Lymph Node Segmentation Training')

parser = argparse.ArgumentParser(description="Configuration parameters for the training script.")

# Define arguments with descriptions, types, and default values
parser.add_argument("--data_dir", type=str, default="../../data/lymph_node/ct_221_0_npz",
                    help="Path to the directory containing the training data.")
parser.add_argument("--model_dir", type=str, default="model_dir_static_lr",
                    help="Directory to save the trained model.")
parser.add_argument("--model_name", type=str, default="SNet",
                    help="Name of the model architecture to use.")
parser.add_argument("--n_fold", type=int, default=5,
                    help="Number of folds for cross-validation.")
parser.add_argument("--fold", type=int, default=3,
                    help="Fold index for cross-validation (0-indexed).")
parser.add_argument("--device_id", type=int, default=0,
                    help="ID of the GPU device to use for training.")
parser.add_argument("--sz", type=int, default=384,
                    help="Input image size.")
parser.add_argument("--num_slices", type=int, default=0,
                    help="Number of slices to use from the 3D volume (0 for all).")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for binary classification.")
parser.add_argument("--dataset", type=str, default="LN Segmentation",
                    help="Name of the dataset to train on.")
parser.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate for the optimizer.")
parser.add_argument("--eps", type=float, default=1e-5,
                    help="Epsilon value for numerical stability.")
parser.add_argument("--weight_decay", type=float, default=1e-5,
                    help="Weight decay parameter for L2 regularization.")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Number of epochs to train the model.")
parser.add_argument("--bs", type=int, default=16,
                    help="Batch size for training.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of steps to accumulate gradients before updating the optimizer.")
parser.add_argument("--SEED", type=int, default=2023,
                    help="Random seed for reproducibility.")
parser.add_argument("--sampling_mode", type=str, default=None,
                    help="Mode of data sampling (e.g., 'upsampling').")
parser.add_argument("--pretrained", action="store_true", default=False,
                    help="Enable using a pre-trained model.")
parser.add_argument("--mixed_precision", action="store_true", default=False,
                    help="Enable mixed precision training (requires compatible hardware).")
parser.add_argument("--resume_path", type=str, default=None,
                    help="Mode of data sampling (e.g., 'upsampling').")
parser.add_argument("--test", action="store_true", default=False,
                    help="Enable using a pre-trained model.")
args = parser.parse_args()

# Create a custom dictionary
config_params = {}
for attr in vars(args):
  config_params[attr] = getattr(args, attr)

# for key, value in config_params.items():
#     if isinstance(value, str):
#         exec(f"{key} = '{value}'")
#     else:
#         exec(f"{key} = {value}")

data_dir = args.data_dir
model_dir = args.model_dir
model_name = args.model_name
n_fold = args.n_fold
fold = args.fold
device_id = args.device_id
sz = args.sz
num_slices = args.num_slices
threshold = args.threshold
dataset = args.dataset
lr = args.lr
eps = args.eps
weight_decay = args.weight_decay
n_epochs = args.n_epochs
bs = args.bs
gradient_accumulation_steps = args.gradient_accumulation_steps
SEED = args.SEED
sampling_mode = args.sampling_mode
pretrained = args.pretrained
mixed_precision = args.mixed_precision
resume_path = args.resume_path
test = args.test

config = vars(args)
# wandb_config = {k: v for k, v in config.items() if k in wandb.sdk.wandb_sdk.INIT_OPTIONS}
print(f'################### Fold:{fold} Training Started ############# \n')
wandb.init(
    project="LN Segmentation",
    config=config_params,
    name=f"{model_name}_fold_{fold}",
    settings=wandb.Settings(start_method='fork')
)

for key, value in color_config.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")

seed_everything(SEED)
df = pd.read_csv(f"{data_dir}/train_labels.csv").drop_duplicates()
train_df = df[(df['fold_patient'] != fold)] 
valid_df = df[df['fold_patient'] == fold]
test_df = pd.read_csv(f"{data_dir}/test_labels.csv").drop_duplicates()
print(len(train_df), len(valid_df), len(test_df))

train_pos = train_df[train_df['label'] == 1] 
train_neg = train_df[train_df['label'] == 0] 

valid_pos = valid_df[valid_df['label'] == 1] 
valid_neg = valid_df[valid_df['label'] == 0] 

test_pos = test_df[test_df['label'] == 1] 
test_neg = test_df[test_df['label'] == 0]

print(f'train:::: pos:{len(train_pos)} neg:{len(train_neg)}')
print(f'valid:::: pos:{len(valid_pos)} neg:{len(valid_neg)}')
print(f'test:::: pos:{len(test_pos)} neg:{len(test_neg)}')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_aug = Augmentation()
train_ds = LNDataset(train_df.path.values, train_df.label.values, dim=sz,
  transforms=train_aug)

valid_ds = LNDataset(valid_df.path.values, valid_df.label.values, dim=sz,
  transforms=None)

test_ds = LNDataset(test_df.path.values, test_df.label.values, dim=sz,
  transforms=None)
sampler = DynamicBalanceClassSampler(labels = train_ds.get_labels(), exp_lambda = 0.95, start_epoch= 5, mode = 'downsampling')
# data = CloudDataset(base_path=data_dir)
# train_ds, valid_ds, test_ds = torch.utils.data.random_split(data, (4000, 2400, 2000))
data_module = DataModule(train_ds, valid_ds, test_ds, batch_size=bs, sampler = sampler)
model = model_params[model_name]
# turn_on_efficient_conv_bn_eval_for_single_model(model)
total_params = sum(p.numel() for p in model.parameters())
wandb.log({'# Model Params': total_params})
flops = FlopCountAnalysis(model, torch.randn(1, 2*num_slices+1, sz, sz))
wandb.log({'# Model FLOPS': flops.total()})
# model = model.to(device)
# device_ids = [device_id]
# print(f'device_ids:{device_ids}')
# model = DataParallel(model, device_ids=device_ids)
# model.to(f'cuda:{device_ids[0]}', non_blocking=True)


## Load Networks
device = torch.device("cuda")
print(f'--- device:{device} ---')
model.to(device)

# citerion = BinaryDiceLoss(reduction='mean')
citerion = FocusNetLoss
plist = [ 
        {'params': model.parameters(),  'lr': lr},
        # {'params': model.head.parameters(),  'lr': lr}
    ]
optim = Adam(plist, lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.5, min_lr=1e-7, verbose=True)
cyclic_scheduler = CosineAnnealingWarmRestarts(optim, 5*len(data_module.train_dataloader()), 2, lr/20, -1)
wandb.watch(models=model, criterion=citerion, log='parameters')

if resume_path:
    best_state = torch.load(f"{model_dir}/fold_{fold}/{resume_path}")
    print(f"Best Validation result was found in epoch {best_state['epoch']}\n")
    print(f"Best Validation Recall {best_state['best_recall']}\n")
    print(f"Best Validation Dice {best_state['best_dice']}\n")
    print("Loading best model")
    prev_epoch_num = best_state['epoch']
    best_valid_loss = best_state['best_loss']
    best_valid_recall = best_state['best_recall']
    best_valid_dice = best_state['best_dice']
    # model.load_state_dict(best_state['model'])
    if list(best_state['model'].keys())[0].startswith('module.'):
        # Create a new state dict without the "module." prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in best_state['model'].items()}
    else:
        new_state_dict = best_state['model']
 
    # Load the adjusted state dict into your model
    model.load_state_dict(new_state_dict)
    optim.load_state_dict(best_state['optim'])
    lr_scheduler.load_state_dict(best_state['scheduler'])
    # cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
else:
    prev_epoch_num = 0
    best_valid_loss = np.inf
    best_valid_recall = 0.0
    best_valid_dice = 0.0
    best_state = None

early_stop_counter = 0
train_losses = []
valid_losses = []
if not test:
    for epoch in range(prev_epoch_num, n_epochs):
        torch.cuda.empty_cache()
        print(gc.collect())
    
        train_loss, train_dice_scores, train_recall_scores, cyclic_scheduler = train_val_class(args, epoch, data_module.train_dataloader(), 
                                                model, citerion, optim, None, mixed_precision=mixed_precision, device_ids=device, train=True)
        valid_loss, val_dice_scores, val_recall_scores, _ = train_val_class(args, epoch, data_module.val_dataloader(), 
                                                model, citerion, optim, None, mixed_precision=mixed_precision, device_ids=device, train=False)
        # NaN check
        if valid_loss != valid_loss:
            print(f'{RED}Mixed Precision{RESET} rendering nan value. Forcing {RED}Mixed Precision{RESET} to be False ...')
            mixed_precision = False
            bs = bs//2
            gradient_accumulation_steps = 2*gradient_accumulation_steps
            print('Loading last best model ...')
            try:
                tmp = torch.load(os.path.join(model_dir, model_name+'_dice_fold_{fold}.pth'))
                model.load_state_dict(tmp['model'])
                optim.load_state_dict(tmp['optim'])
                lr_scheduler.load_state_dict(tmp['scheduler'])
                # cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
                del tmp
            except:
                model = model_params[model_name]
                model = model.to(device)
        else:
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        
        # lr_scheduler.step(valid_loss)
        lr_scheduler.step(np.mean(val_dice_scores))
        wandb.log({"Train Average DICE": np.mean(train_dice_scores), "Train SD DICE": np.std(train_dice_scores), "Epoch": epoch})
        wandb.log({"Validation Average DICE": np.mean(val_dice_scores), "Validation SD DICE": np.std(val_dice_scores),"Epoch": epoch})
        wandb.log({"Train Average Recall": np.mean(train_recall_scores), "Train SD Recall": np.std(train_recall_scores), "Epoch": epoch})
        wandb.log({"Validation Average Recall": np.mean(val_recall_scores), "Validation SD Recall": np.std(val_recall_scores),"Epoch": epoch})
        
        print(ITALIC+"="*70+RESET)
        print(f"{BOLD}{UNDERLINE}{CYAN}Epoch {epoch} Report:{RESET}")
        print(f"{MAGENTA}Validation Loss: {valid_loss :.4f} Dice Score: {np.mean(val_dice_scores) :.4f} Recall Score: {np.mean(val_recall_scores) :.4f}{RESET}")
        model_dict = {'model': model.state_dict(), 
        'optim': optim.state_dict(), 
        'scheduler':lr_scheduler.state_dict(), 
        # 'cyclic_scheduler':cyclic_scheduler.state_dict(), 
        # 'scaler': scaler.state_dict(),
        'best_loss':valid_loss, 
        'best_recall':np.mean(val_recall_scores),
        'best_dice':np.mean(val_dice_scores),
        'epoch':epoch}
        if np.mean(val_dice_scores) < best_valid_dice:
            early_stop_counter += 1
            if early_stop_counter == 30:
                print(f"{RED}No improvement over val recall for so long!{RESET}")
                print(f"{RED}Early Stopping now!{RESET}")
                break
        else: early_stop_counter = 0
    
        best_valid_dice, best_state = save_model(np.mean(val_dice_scores), 
                    best_valid_dice, model_dict, 
                    model_name, f"{model_dir}/fold_{fold}", 'dice', epoch, fold, 'max')
        print(f'best dice:{best_valid_dice}, epoch:{epoch} fold:{fold}')
        
        print(ITALIC+"="*70+RESET)

print(f"########### Testing: fold:{fold} ##############")
# Dude, your best model saving way is weird.
best_model_path = f"{model_dir}/fold_{fold}/{model_name}_dice_fold_{fold}.pth"
print(f'best model path:{best_model_path}')
best_state = torch.load(best_model_path)
model.load_state_dict(best_state['model'])
optim.load_state_dict(best_state['optim'])
lr_scheduler.load_state_dict(best_state['scheduler'])
# cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
print(f"{BLUE}Best Validation result was found in epoch {best_state['epoch']}\n{RESET}")
print(f"{BLUE}Best Validation Recall {best_state['best_recall']}\n{RESET}")
print(f"{BLUE}Best Validation Dice {best_state['best_dice']}\n{RESET}")
epoch = 0
test_loss, test_dice_scores, test_recall_scores, _ = train_val_class(args, epoch, data_module.test_dataloader(), 
                                            model, citerion, optim, None, mixed_precision=mixed_precision, device_ids=device, train=False)
wandb.log({"Test Loss": test_loss, "Test Average DICE": np.mean(test_dice_scores), "Test SD DICE": np.std(test_dice_scores), "Test Average Recall": np.mean(test_recall_scores), "Test SD Recall": np.std(test_recall_scores)})
wandb.finish()