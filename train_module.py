from collections import OrderedDict
import time
import cv2
import numpy as np
import torch
import wandb
from torch.nn import functional as F

from config.config import config_params
from config.color_config import color_config
from utils import clip_gradient, visualizer

from metric import dice_coefficient_one_class, dice_score_by_data_torch, recall
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
    pred = torch.squeeze(pred)
    mask = torch.squeeze(mask)
    mask = mask.to(dtype=torch.float32)
    avg_pooling = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    neg_part_base = 1
    
    #omitting
    weit =  neg_part_base + 5*avg_pooling                                                   
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*bce)
    wbce = wbce.sum(dim=(1, 2))/weit.sum(dim=(1, 2))
    
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(1, 2))
    union = ((pred + mask)*weit).sum(dim=(1, 2))
    wiou = 1 - ((inter + 1)/(union - inter+1))
    
    m_wbce = wbce.mean()
    m_iou = wiou.mean()

    return m_wbce, m_iou

def train_val_seg(epoch, dataloader, model, criterion, optimizer, cyclic_scheduler=None, run_id=0, mixed_precision=False, device_ids=[0], train=True):
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
    partial_map = False
    heatmap_prediction = False
    plot_img = 10
    
    scaler = torch.cuda.amp.GradScaler()
    stage = 'train' if train else 'validation'

    model.train() if train else model.eval()
    print(f"{BLUE}Initiating {stage} phase ...{RESET}")
    for idx, (data, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(train):
            data = data.to(f'cuda:{device_ids[0]}', non_blocking=True).to(dtype=torch.float32)
            labels = labels.to(f'cuda:{device_ids[0]}', non_blocking=True).to(dtype=torch.long)
            epoch_samples += len(data)

            with torch.cuda.amp.autocast(mixed_precision):
<<<<<<< HEAD
                outputs = model(data)
                loss = criterion(labels, outputs)
=======
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
                
>>>>>>> 380b29f2829d74a26fff6c0d556fc41b22b87f56
                running_loss += loss.item()*len(data)

                wbce_loss.append(wbce.item())
                wiou_loss.append(wiou.item())
                focal_losses.append(fl.item())
                tversky_losses.append(dl.item())
                losses.append(loss.item())

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
<<<<<<< HEAD
            if cyclic_scheduler is not None: cyclic_scheduler.step()
            elapsed = int(time.time() - t1)
            eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
            if isinstance(outputs, tuple): 
                outputs = outputs[0]
                # print(torch.max(outputs), torch.min(outputs))
                # outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)

            dice_scores_batch = dice_score_by_data_torch(labels, outputs, threshold = threshold).detach().cpu().numpy()
            recall_score_batch = recall(labels, outputs, threshold = threshold).detach().cpu().numpy()
            # print("Recall", (recall_score_batch))
            # Find the index of the image with the lowest loss
            min_recall_index = np.argmin(recall_score_batch)
            # To avoid recall = 1 cases where the mask and predictions are all black
            recall_score_batch_ = [i if i<1 else -100 for i in recall_score_batch]
            max_recall_index = np.argmax(recall_score_batch_)
            average_recall = np.mean(recall_score_batch)
            closest_to_average_index = np.argmin(np.abs(recall_score_batch - average_recall))
            if not train:
                row1 = visualizer(data[:, num_slices, :, :], outputs, labels, min_recall_index, recall_score_batch)
                row2 = visualizer(data[:, num_slices, :, :], outputs, labels, closest_to_average_index, recall_score_batch)
                row3 = visualizer(data[:, num_slices, :, :], outputs, labels, max_recall_index, recall_score_batch)
                final_image = np.vstack([row1, row2, row3])
                # print(final_image.shape, row1.shape)
                wandb.log({f"image_batch {idx}": wandb.Image(final_image)})
            dice_scores.extend(dice_scores_batch)
            recall_scores.extend(recall_score_batch)
            current_loss = running_loss / epoch_samples
            msg = f'{ITALIC}{PURPLE}Epoch: {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {current_loss:.4f} Time: {elapsed}s ETA: {eta} s{RESET}' if train else f'{ITALIC}{RED}Epoch {epoch+1} Progress: [{idx}/{len(dataloader)}] loss: {current_loss:.4f} Time: {elapsed}s ETA: {eta} s{RESET}'
            wandb.log({"Train Loss" if train else "Validation Loss": current_loss, "Epoch": epoch})
            print(msg, end='\r')
    print(f'{stage} Loss: {running_loss/epoch_samples:.4f}')
    return running_loss / epoch_samples, dice_scores, recall_scores, cyclic_scheduler
=======
                ### FocusNet
                if outputs.shape[1]>1:
                    predictions = torch.nn.functional.softmax(outputs, dim=1)
                    pred_labels = torch.argmax(predictions, dim=1) 
                    pred_labels = pred_labels.float()
                    outputs_prob = torch.unsqueeze(pred_labels, dim=1)
                else:
                    outputs_prob = (torch.sigmoid(outputs)>threshold).float()
                    if partial_map:
                        pd_outputs = (torch.sigmoid(lateral_map_1)>threshold).float()

                #####plotting###########
            #     if idx%plot_img==0:
            #         if partial_map:
            #             visuals = OrderedDict([('input', data[0:8, :, :, :]),
            #                                     ('mask', labels[0:8, :, :, :]),
            #                                     ('output', outputs[0:8, :, :, :]),
            #                                     ('partial_d', pd_outputs[0:8, :, :, :])])
            #         else:
            #             if not heatmap_prediction:
            #                 visuals = OrderedDict([('input', data[0:8, :, :, :]),
            #                                     ('mask', labels[0:8, :, :]),
            #                                     ('output', outputs_prob[0:8, :, :, :])])
            #             else:
            #                 visuals = OrderedDict([('input', data[0:8, :, :, :]),
            #                                 ('mask', labels[0:8, :, :]),
            #                                 ('output', outputs_prob[0:8, :, :, :]),
            #                                 ('gt_hmap', heatmaps[0:8, :, :, :]),
            #                                 ('p_hmap', h_preds[0:8, :, :, :])])
            # if train:
            #     write_img(visuals, run_id, epoch, idx)
            # else:
            #     write_img(visuals, run_id, epoch, idx, val=True)

            dice_val = dice_score_by_data_torch(labels, outputs_prob, threshold=threshold)
            
            dice_scores.append(dice_val) ###[]
            
            dice_running_avg = torch.mean(torch.cat(dice_scores).cpu()).item()
            msg = f'Epoch: {epoch} Progress: [{idx}/{len(dataloader)}] loss: {(np.mean(losses)):.4f} Dice {dice_running_avg:.4f} f_l:{np.mean(focal_losses):.4f} t_l:{np.mean(tversky_losses):.4f} wbce_l:{np.mean(wbce_loss):.4f}  wiou_l:{np.mean(wiou_loss):.4f}'
            
            # if train:
            #     display = display_train_value
            # else:
            #     display = display_valid_value
            display = 10
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
>>>>>>> 380b29f2829d74a26fff6c0d556fc41b22b87f56
