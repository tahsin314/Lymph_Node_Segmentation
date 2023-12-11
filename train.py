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
# from catalyst.data.sampler import BalanceClassSampler
from thop import profile

import wandb
from Datasets.CloudDataset import CloudDataset
from Datasets.DataModule import DataModule

from config import config_params, model_params, aug_config, color_config
from Datasets.LNDataset import LNDataset
from train_module import train_val_class
from utils import save_model, seed_everything
from losses.tversky import tversky_loss, focal_tversky
from losses.dice import dice_loss, dice_lossv2
from losses.loss import FocusNetLoss, DiceLoss
from losses.hybrid import hybrid_loss
from losses.structure_loss import structure_loss, total_structure_loss

from models.utils import turn_on_efficient_conv_bn_eval_for_single_model

wandb.init(
    project="LN Segmentation",
    config=config_params,
    name=f"{config_params['model_name']}",
    settings=wandb.Settings(start_method='fork')
)

for key, value in config_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")

for key, value in color_config.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
        
seed_everything(SEED)
df = pd.read_csv(f"{data_dir}/train_labels.csv")
train_df = df[(df['fold_patient'] != fold)] 
valid_df = df[df['fold_patient'] == fold]
test_df = pd.read_csv(f"{data_dir}/test_labels.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_ds = LNDataset(train_df.path.values, train_df.label.values, dim=sz,
  transforms=None)

valid_ds = LNDataset(valid_df.path.values, valid_df.label.values, dim=sz,
  transforms=None)

test_ds = LNDataset(test_df.path.values, test_df.label.values, dim=sz,
  transforms=None)

# data = CloudDataset(base_path=data_dir)
# train_ds, valid_ds, test_ds = torch.utils.data.random_split(data, (4000, 2400, 2000))
data_module = DataModule(train_ds, valid_ds, test_ds, batch_size=bs)
model = model_params[config_params['model_name']]
# turn_on_efficient_conv_bn_eval_for_single_model(model)
total_params = sum(p.numel() for p in model.parameters())
wandb.log({'# Model Params': total_params})
flops, _ = profile(model, inputs=(torch.randn(1, num_channels, sz, sz),))
wandb.log({'# Model FLOPS': flops})
# model = model.to(device)
device_ids = [1, 0, 2, 3]
model = DataParallel(model, device_ids=device_ids)
model.to(f'cuda:{device_ids[0]}', non_blocking=True)

# citerion = BinaryDiceLoss(reduction='mean')
citerion = structure_loss
plist = [ 
        {'params': model.parameters(),  'lr': lr},
        # {'params': model.head.parameters(),  'lr': lr}
    ]
optim = Adam(plist, lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)
cyclic_scheduler = CosineAnnealingWarmRestarts(optim, 5*len(data_module.train_dataloader()), 2, lr/20, -1)
wandb.watch(models=model, criterion=citerion, log='parameters')

if pretrained:
    best_state = torch.load(f"model_dir/{model_name}_loss.pth")
    print(f"Best Validation result was found in epoch {best_state['epoch']}\n")
    print(f"Best Validation Loss {best_state['best_loss']}\n")
    print("Loading best model")
    prev_epoch_num = best_state['epoch']
    best_valid_loss = best_state['best_loss']
    model.load_state_dict(best_state['model'])
    optim.load_state_dict(best_state['optim'])
    lr_scheduler.load_state_dict(best_state['scheduler'])
    # cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
else:
    prev_epoch_num = 0
    best_valid_loss = np.inf
    best_state = None

early_stop_counter = 0
train_losses = []
valid_losses = []

for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())

    train_loss, train_dice_scores, train_recall_scores, cyclic_scheduler = train_val_class(epoch, data_module.train_dataloader(), 
                                            model, citerion, optim, cyclic_scheduler, mixed_precision=mixed_precision, device_ids=device_ids, train=True)
    valid_loss, val_dice_scores, val_recall_scores, _ = train_val_class(epoch, data_module.val_dataloader(), 
                                            model, citerion, optim, cyclic_scheduler, mixed_precision=mixed_precision, device_ids=device_ids, train=False)
    # NaN check
    if valid_loss != valid_loss:
        print(f'{RED}Mixed Precision{RESET} rendering nan value. Forcing {RED}Mixed Precision{RESET} to be False ...')
        mixed_precision = False
        bs = bs//2
        gradient_accumulation_steps = 2*gradient_accumulation_steps
        print('Loading last best model ...')
        try:
            tmp = torch.load(os.path.join(model_dir, model_name+'_loss.pth'))
            model.load_state_dict(tmp['model'])
            optim.load_state_dict(tmp['optim'])
            lr_scheduler.load_state_dict(tmp['scheduler'])
            cyclic_scheduler.load_state_dict(tmp['cyclic_scheduler'])
            del tmp
        except:
            model = model_params[config_params['model_name']]
            model = model.to(device)
    else:
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    
    # lr_scheduler.step(valid_loss)
    lr_scheduler.step(valid_loss)
    wandb.log({"Train Average DICE": np.mean(train_dice_scores), "Train SD DICE": np.std(train_dice_scores), "Epoch": epoch})
    wandb.log({"Validation Average DICE": np.mean(val_dice_scores), "Validation SD DICE": np.std(val_dice_scores),"Epoch": epoch})
    wandb.log({"Train Average Recall": np.mean(train_recall_scores), "Train SD DICE": np.std(train_recall_scores), "Epoch": epoch})
    wandb.log({"Validation Average Recall": np.mean(val_recall_scores), "Validation SD DICE": np.std(val_recall_scores),"Epoch": epoch})
    
    print(ITALIC+"="*70+RESET)
    print(f"{BOLD}{UNDERLINE}{CYAN}Epoch {epoch+1} Report:{RESET}")
    print(f"{MAGENTA}Validation Loss: {valid_loss :.4f} Dice Score: {np.mean(val_dice_scores) :.4f} Recall Score: {np.mean(val_recall_scores) :.4f}{RESET}")
    model_dict = {'model': model.state_dict(), 
    'optim': optim.state_dict(), 
    'scheduler':lr_scheduler.state_dict(), 
    'cyclic_scheduler':cyclic_scheduler.state_dict(), 
    # 'scaler': scaler.state_dict(),
    'best_loss':valid_loss, 
    'epoch':epoch}
    if valid_loss > best_valid_loss:
        early_stop_counter += 1
        if early_stop_counter == 30:
            print(f"{RED}No improvement over val loss for so long!{RESET}")
            print(f"{RED}Early Stopping now!{RESET}")
            break
    else: early_stop_counter = 0

    best_valid_loss, best_state = save_model(valid_loss, 
                best_valid_loss, model_dict, 
                model_name, 'model_dir')
    
    print(ITALIC+"="*70+RESET)
    
best_state = torch.load(f"model_dir/{model_name}_loss.pth")
model.load_state_dict(best_state['model'])
optim.load_state_dict(best_state['optim'])
lr_scheduler.load_state_dict(best_state['scheduler'])
# cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
print(f"{BLUE}Best Validation result was found in epoch {best_state['epoch']}\n{RESET}")
print(f"{BLUE}Best Validation Loss {best_state['best_loss']}\n{RESET}")
test_loss, test_dice_scores, test_recall_scores, _ = train_val_class(epoch, data_module.test_dataloader(), 
                                            model, citerion, optim, cyclic_scheduler, mixed_precision=mixed_precision, device_ids=device_ids, train=False)
wandb.log({"Test Loss": test_loss, "Test Average DICE": np.mean(test_dice_scores), "Test SD DICE": np.std(test_dice_scores), "Test Average Recall": np.mean(test_recall_scores), "Test SD Recall": np.std(test_recall_scores)})
wandb.finish()