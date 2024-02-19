import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
from datetime import datetime
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

from config.config import config_params
from config.model_config import model_params
from config.augment_config import aug_config
from config.color_config import color_config
from Datasets.LNDataset import LNDataset
from train_module import train_val_seg
from utils import save_model, seed_everything
from losses.tversky import tversky_loss, focal_tversky
from losses.dice import dice_loss, dice_lossv2
from losses.focusnetloss import FocusNetLoss
from losses.hybrid import hybrid_loss
from losses.structure_loss import structure_loss, total_structure_loss

from fvcore.nn import FlopCountAnalysis

from models.utils import turn_on_efficient_conv_bn_eval_for_single_model
from models.Focus_Net.s_net import s_net
from losses.losses import FocusNetLoss

from tensorboardX import SummaryWriter

wandb.init(
    project="TC Segmentation",
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
df = pd.read_csv(f"{data_dir}/train_labels.csv").drop_duplicates()
train_df = df[(df['fold_patient'] != fold)] 
valid_df = df[df['fold_patient'] == fold]
test_df = pd.read_csv(f"{data_dir}/test_labels.csv").drop_duplicates()
print(len(train_df), len(valid_df), len(test_df))

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
# model = model_params[config_params['model_name']]

model = s_net(1, 1)
# turn_on_efficient_conv_bn_eval_for_single_model(model)
total_params = sum(p.numel() for p in model.parameters())
wandb.log({'# Model Params': total_params})
flops = FlopCountAnalysis(model, torch.randn(1, 2*num_slices+1, sz, sz))
<<<<<<< HEAD
=======
# flops, _ = profile(model, inputs=(torch.randn(1, 2*num_slices+1, sz, sz),))
>>>>>>> 380b29f2829d74a26fff6c0d556fc41b22b87f56
wandb.log({'# Model FLOPS': flops.total()})
# model = model.to(device)
device_ids = [1, 0, 2, 3]
model = DataParallel(model, device_ids=device_ids)
model.to(f'cuda:{device_ids[0]}', non_blocking=True)

# citerion = BinaryDiceLoss(reduction='mean')
<<<<<<< HEAD
citerion = FocusNetLoss
=======
criterion = FocusNetLoss()	# Binary focal and tversky loss
>>>>>>> 380b29f2829d74a26fff6c0d556fc41b22b87f56
plist = [ 
        {'params': model.parameters(),  'lr': lr},
        # {'params': model.head.parameters(),  'lr': lr}
    ]
optim = Adam(plist, lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, mode='max', patience=5, factor=0.5, min_lr=1e-6, verbose=True)
cyclic_scheduler = CosineAnnealingWarmRestarts(optim, 5*len(data_module.train_dataloader()), 2, lr/20, -1)
wandb.watch(models=model, criterion=criterion, log='parameters')

if pretrained:
    best_state = torch.load(f"model_dir/{model_name}_dice.pth")
    print(f"Best Validation result was found in epoch {best_state['epoch']}\n")
    print(f"Best Validation Recall {best_state['best_recall']}\n")
    print(f"Best Validation Dice {best_state['best_dice']}\n")
    print("Loading best model")
    prev_epoch_num = best_state['epoch']
    best_valid_loss = best_state['best_loss']
    best_valid_recall = best_state['best_recall']
    best_valid_dice = best_state['best_dice']
    model.load_state_dict(best_state['model'])
    optim.load_state_dict(best_state['optim'])
    lr_scheduler.load_state_dict(best_state['scheduler'])
    cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
else:
    prev_epoch_num = 0
    best_valid_loss = np.inf
    best_valid_recall = 0.0
    best_valid_dice = 0.0
    best_state = None

early_stop_counter = 0
train_losses = []
valid_losses = []

log_dir = './history_dir/logs'
os.makedirs(log_dir, exist_ok=True)
run_id = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
writer = SummaryWriter(os.path.join(log_dir, str(run_id)))

for epoch in range(prev_epoch_num, n_epochs):
    torch.cuda.empty_cache()
    print(gc.collect())

    train_loss, train_dice, train_fl, train_tl, t_wbce_l, t_wiou_l, cyclic_scheduler, model = train_val_seg(epoch, data_module.train_dataloader(), model, criterion, optim, run_id=run_id, mixed_precision=mixed_precision, train=True, device_ids=device_ids)
    valid_loss, valid_dice, valid_fl, valid_tl, v_wbce_l, v_wiou_l,  = train_val_seg(epoch, data_module.val_dataloader(), model, criterion, optim, run_id=run_id, mixed_precision=mixed_precision, train=False, device_ids=device_ids)
    # NaN check
    if valid_loss != valid_loss:
        print(f'{RED}Mixed Precision{RESET} rendering nan value. Forcing {RED}Mixed Precision{RESET} to be False ...')
        mixed_precision = False
        bs = bs//2
        gradient_accumulation_steps = 2*gradient_accumulation_steps
        print('Loading last best model ...')
        try:
            tmp = torch.load(os.path.join(model_dir, model_name+'_dice.pth'))
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

    writer.add_scalar('Training Total Loss', train_loss, epoch)
    writer.add_scalar('Train wBCE Loss', t_wbce_l, epoch)
    writer.add_scalar('Train wiou Loss', t_wiou_l, epoch)
    writer.add_scalar('Train Focal Loss', train_fl, epoch)
    writer.add_scalar('Train Tversky Loss', train_tl, epoch)
    writer.add_scalar('Training Avg D_Coeff', train_dice, epoch)

    writer.add_scalar('Valid Total Loss', valid_loss, epoch)
    writer.add_scalar('Validation wBCE Loss', v_wbce_l, epoch)
    writer.add_scalar('Valid wiou Loss', v_wiou_l, epoch)
    writer.add_scalar('Validation Focal Loss', valid_fl, epoch)
    writer.add_scalar('Validation Tversky Loss', valid_tl, epoch)
    writer.add_scalar('Validation Avg D_Coeff', valid_dice, epoch)
    
    # lr_scheduler.step(valid_loss)
    lr_scheduler.step(np.mean(val_dice_scores))
    wandb.log({"Train Average DICE": np.mean(train_dice_scores), "Train SD DICE": np.std(train_dice_scores), "Epoch": epoch})
    wandb.log({"Validation Average DICE": np.mean(val_dice_scores), "Validation SD DICE": np.std(val_dice_scores),"Epoch": epoch})
    wandb.log({"Train Average Recall": np.mean(train_recall_scores), "Train SD Recall": np.std(train_recall_scores), "Epoch": epoch})
    wandb.log({"Validation Average Recall": np.mean(val_recall_scores), "Validation SD Recall": np.std(val_recall_scores),"Epoch": epoch})
    
    print(ITALIC+"="*70+RESET)
    print(f"{BOLD}{UNDERLINE}{CYAN}Epoch {epoch+1} Report:{RESET}")
    print(f"{MAGENTA}Validation Loss: {valid_loss :.4f} Dice Score: {np.mean(val_dice_scores) :.4f} Recall Score: {np.mean(val_recall_scores) :.4f}{RESET}")
    model_dict = {'model': model.state_dict(), 
    'optim': optim.state_dict(), 
    'scheduler':lr_scheduler.state_dict(), 
    'cyclic_scheduler':cyclic_scheduler.state_dict(), 
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
                model_name, 'model_dir', 'dice', 'max')
    
    print(ITALIC+"="*70+RESET)

torch.cuda.empty_cache()
best_state = torch.load(f"model_dir/{model_name}_dice.pth")
model.load_state_dict(best_state['model'])
optim.load_state_dict(best_state['optim'])
lr_scheduler.load_state_dict(best_state['scheduler'])
# cyclic_scheduler.load_state_dict(best_state['cyclic_scheduler'])
print(f"{BLUE}Best Validation result was found in epoch {best_state['epoch']}\n{RESET}")
print(f"{BLUE}Best Validation Recall {best_state['best_recall']}\n{RESET}")
print(f"{BLUE}Best Validation Dice {best_state['best_dice']}\n{RESET}")
test_loss, test_dice_scores, test_recall_scores, _ = train_val_class(-1, data_module.test_dataloader(), 
                                            model, citerion, optim, cyclic_scheduler, mixed_precision=mixed_precision, device_ids=device_ids, train=False)
wandb.log({"Test Loss": test_loss, "Test Average DICE": np.mean(test_dice_scores), "Test SD DICE": np.std(test_dice_scores), "Test Average Recall": np.mean(test_recall_scores), "Test SD Recall": np.std(test_recall_scores)})
wandb.finish()