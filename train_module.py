import time
import cv2
import numpy as np
import torch
import wandb

from config.config import config_params
from config.color_config import color_config
from utils import clip_gradient, visualizer

from metric import dice_score_by_data_torch, recall
from visualizer import write_img

for key, value in color_config.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")

for key, value in config_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")

def structure_loss(pred, mask):
	avg_pooling = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	neg_part_base = 1
	
	#omitting
	weit =  neg_part_base + 5*avg_pooling  
														
	bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
	wbce = (weit*bce)
	wbce = wbce.sum(dim=(2, 3))/weit.sum(dim=(2, 3))
	
	pred = torch.sigmoid(pred)
	inter = ((pred * mask)*weit).sum(dim=(2, 3))
	union = ((pred + mask)*weit).sum(dim=(2, 3))
	wiou = 1 - ((inter + 1)/(union - inter+1))
	
	m_wbce = wbce.mean()
	m_iou = wiou.mean()

	return m_wbce, m_iou

def train_val_class(epoch, dataloader, model, criterion, optimizer, cyclic_scheduler, mixed_precision=False, device_ids=[0], train=True):
    t1 = time.time()
    running_loss = 0
	epoch_samples = 0
    
    dice_scores = []
	raw_dice_coeff = 0
	raw_val_dice = 0

	wbce_loss = []
	wiou_loss = []
	losses = []
	focal_losses = []
	tversky_losses = []
	
	
    scaler = torch.cuda.amp.GradScaler()
    stage = 'train' if train else 'validation'

    model.train() if train else model.eval()
    print(f"{BLUE}Initiating {stage} phase ...{RESET}")
    for idx, (data, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(train):
            data = data.to(f'cuda:{device_ids[0]}', non_blocking=True).to(dtype=torch.float32)
            labels = labels.to(f'cuda:{device_ids[0]}', non_blocking=True).to(dtype=torch.long)
            # print(data.max(), data.min(), labels.max(), labels.min())
            # print(data.size(), labels.size())
            epoch_samples += len(data)

            with torch.cuda.amp.autocast(mixed_precision):
                # outputs = model(data)
                outputs, lateral_map_2, lateral_map_1 = model(data)
                # print(outputs.max(), outputs.min())
                fl, dl = criterion(outputs, labels)
                wbce2, wiou2 = structure_loss(lateral_map_2, labels)
                wbce1, wiou1 = structure_loss(lateral_map_1, labels)
                # h_loss = torch.nn.functional.mse_loss(h_preds, heatmaps)
                wbce = wbce2+wbce1
                wiou = wiou2+wiou1
                
                loss = dl + 2*wbce + wiou
                
                running_loss += loss.item()*len(data)

                wbce_loss.append(wbce.item())
                wiou_loss.append(wiou.item())
                focal_losses.append(fl.item())
                tversky_losses.append(dl.item())

                if train:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        clip_gradient(optimizer, 25)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else:
                        loss.backward()
                        clip_gradient(optimizer, 25)
                        optimizer.step()
                        optimizer.zero_grad()
                ### FocusNet
                if outputs.shape[1]>1:
                    predictions = torch.nn.functional.softmax(outputs, dim=1)
                    pred_labels = torch.argmax(predictions, dim=1) 
                    pred_labels = pred_labels.float()
                    outputs_prob = torch.unsqueeze(pred_labels, dim=1)
                else:
                    outputs_prob = (torch.sigmoid(outputs)>dice_threshold).float()
                    if partial_map:
                        pd_outputs = (torch.sigmoid(lateral_map_1)>dice_threshold).float()

                #####plotting###########
                if idx%plot_img==0:
                    if partial_map:
                        visuals = OrderedDict([('input', inp_8_bit[0:8, :, :, :]),
                                                ('mask', labels[0:8, :, :, :]),
                                                ('output', outputs[0:8, :, :, :]),
                                                ('partial_d', pd_outputs[0:8, :, :, :])])
                    else:
                        if not heatmap_prediction:
                            visuals = OrderedDict([('input', inp_8_bit[0:8, :, :, :]),
                                                ('mask', labels[0:8, :, :, :]),
                                                ('output', outputs_prob[0:8, :, :, :])])
                        else:
                            visuals = OrderedDict([('input', inp_8_bit[0:8, :, :, :]),
                                            ('mask', labels[0:8, :, :, :]),
                                            ('output', outputs_prob[0:8, :, :, :]),
                                            ('gt_hmap', heatmaps[0:8, :, :, :]),
                                            ('p_hmap', h_preds[0:8, :, :, :])])
            if train:
                write_img(visuals, run_id, epoch, idx)
            else:
                write_img(visuals, run_id, epoch, idx, val=True)

            dice_val = dice_coeff(outputs_prob, labels, threshold=None)
            dice_scores.append(dice_val) ###[]
            
            dice_running_avg = torch.mean(torch.cat(dice_scores).cpu()).item()
            msg = f'Epoch: {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(np.mean(losses)):.4f} Dice {dice_running_avg:.4f} f_l:{np.mean(focal_losses):.4f} t_l:{np.mean(tversky_losses):.4f} wbce_l:{np.mean(wbce_loss):.4f}  wiou_l:{np.mean(wiou_loss):.4f}'
            
            if train:
                display = display_train_value
            else:
                display = display_valid_value
            if idx%display==0:
                print(msg)


    if cyclic_scheduler is not None: cyclic_scheduler.step()
    elapsed = int(time.time() - t1)
    eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
    mean_dice_scores = torch.mean(torch.cat(dice_scores)).cpu().item()
	
    if train: stage='train'
	else: stage='validation'

	# Asymmetric FL and TFL loss
	avg_fl = np.mean(focal_losses)
	avg_tl = np.mean(tversky_losses)
	######
	# WBCE and WIOU Loss
	avg_wbce_l = np.mean(wbce_loss)
	avg_wiou_l = np.mean(wiou_loss)
	#########
	avg_hloss = np.mean(heatmap_losses)
	avg_loss = np.mean(losses)

	
	msg = f'stage:{stage} loss: {avg_loss:.4f} Dice {mean_dice_scores:.4f} f_l:{avg_fl:.4f} tl:{avg_tl:.4f} wbce_l:{avg_wbce_l:.4f}  wiou_l:{avg_wiou_l:.4f} h_loss:{avg_hloss}'
	
	print(msg)
	if train:
		return avg_loss, mean_dice_scores, avg_fl, avg_tl, avg_wbce_l, avg_wiou_l, cyclic_scheduler, model
	else:
		return avg_loss, mean_dice_scores, avg_fl, avg_tl, avg_wbce_l, avg_wiou_l