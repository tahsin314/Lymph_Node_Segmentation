import os

from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import wandb
from config.config import *
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import cv2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm as T
import pandas as pd
from torch import nn
from config.color_config import color_config
# from gradcam.gradcam import GradCAM, GradCAMpp

for key, value in color_config.items():
    if isinstance(value, str):
        exec(f"{key} = '{value}'")
        
def onehot_encoder(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def window_image(img, window_center=40, window_width=400, intercept=0, slope=1, rescale=True):
    # transform to hu
    # 16 bit
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    
    # [-160, 240] --> 400 ... signed 8 bit --> [-127, 128]
    img_min = window_center - window_width//2 #minimum HU level -160
    img_max = window_center + window_width//2 #maximum HU level  240
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    #print('img data type: ',img.dtype)

    # 16 bit-->[0, 1]
    img_norm = (img - img_min) / (img_max - img_min)
    #print('img_norm type: ',img_norm.dtype)
    
    # 8-bit: [0, 255] -->int 8 bit
    img_8_bit = (img_norm*255.0).astype('uint8')
    return img_8_bit, img_norm

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def model_summary(model, sample_input):
    # Initialize variables to collect layer information
    layers_info = []
    x = sample_input
    for name, layer in model.named_children():
        input_shape = tuple(x.shape) if x is not None else None
        if isinstance(layer, nn.Linear):
            x = x.view(x.size(0), -1)  # Example reshape operation
        x = layer(x)
        output_shape = tuple(x.shape) if hasattr(layer, 'weight') else None
        num_params = sum(p.numel() for p in layer.parameters())
        layers_info.append([name, input_shape, output_shape, num_params])

    # Compute the total number of parameters
    total_params = sum(num_params for _, _, _, num_params in layers_info)
    columns=["Layer Name", "Input Shape", "Output Shape", "Param #"]
    # Create a Pandas DataFrame
    df = pd.DataFrame(layers_info, columns=columns)
    return df, columns, total_params

def save_model(valid_metric, best_valid_metric, model_dict, model_name, save_dir, metric_name, mode= 'max'):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mode_dict = {'max' :  'increase', 'min' : 'decrease'}
    if mode == 'max':
        if valid_metric>best_valid_metric:
            print(f'{BOLD}{GREEN}Validation {metric_name} has {mode_dict[mode]}d from:  {best_valid_metric:.4f} to: {valid_metric:.4f}{RESET}')
            best_valid_metric = valid_metric
            save_file_path = os.path.join(save_dir, f'{model_name}_{metric_name}.pth')
            torch.save(model_dict, save_file_path)
    
    return best_valid_metric, model_dict 

def visualizer(predictions, outputs, labels, idx, metric_scores_batch):
    data = predictions[idx].detach().cpu().numpy()
    if data.shape[0] == 1: data = np.squeeze(data, axis=0)
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    if data.ndim == 3:
        data = data[0, :, :]
        # data = data.transpose(1, 2, 0)
        data = 255.*cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)
    else: data = 255.*cv2.cvtColor(data,cv2.COLOR_GRAY2RGB)
    
    output = 255.*outputs[idx].sigmoid().detach().cpu().numpy()[0]
    output = output.astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    label = 255.*labels[idx].cpu().numpy()
    label = label.astype(np.uint8)
    label = cv2.cvtColor(label ,cv2.COLOR_GRAY2RGB)
    # Create a blank image for the text
    text_image = np.zeros((80, data.shape[1] * 3, 3), dtype=np.uint8)

    # Add Input, Prediction, Output texts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)

    cv2.putText(text_image, "Input", (10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(text_image, "Prediction", (data.shape[1] + 10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.putText(text_image, "Output", (2 * data.shape[1] + 10, 40), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Add metric score below Prediction text
    cv2.putText(text_image, f"Metric Score: {metric_scores_batch[idx]:.3f}", (data.shape[1] + 10, 70), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    # print(metric_scores_batch[idx])
    # Combine images horizontally with text
    result_image = np.vstack([text_image, np.hstack([data, output, label])])

    return result_image