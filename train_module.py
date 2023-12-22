import time
import cv2
import numpy as np
import torch
import wandb

from config import color_config, config_params
from utils import clip_gradient, visualizer

from metric import dice_score_by_data_torch, recall

for key, value in color_config.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")

for key, value in config_params.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
    else:
        exec(f"{key} = {value}")

def train_val_class(epoch, dataloader, model, criterion, optimizer, cyclic_scheduler, mixed_precision=False, device_ids=[0], train=True):
    t1 = time.time()
    running_loss = 0
    epoch_samples = 0
    dice_scores = []
    recall_scores = []
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
                outputs = model(data)
                # print(outputs.max(), outputs.min())
                loss = criterion(labels, outputs)
                running_loss += loss.item()*len(data)

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
            if cyclic_scheduler is not None: cyclic_scheduler.step()
            elapsed = int(time.time() - t1)
            eta = int(elapsed / (idx+1) * (len(dataloader)-(idx+1)))
            if isinstance(outputs, tuple): 
                outputs = outputs[0]
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