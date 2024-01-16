from tensorboardX import SummaryWriter
import os
import torch
import torchvision
import config as cfg
import numpy as np
import cv2
import math


def draw_contour(img, mask, color=(0, 0, 255), thickness=1):
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    #img = np.ascontiguousarray(img, dtype=np.uint8)
    if len(img.shape)==2:
        img = np.stack((img,)*3, axis=-1)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img


def write_img(visuals, run_id, ep, iteration, val=False, test=False, caranet = False):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    if not val:
        if not test:
            path = os.path.join(cfg.output_dir, run_id, 'train', str(ep))
        else:
            folder_id = math.floor(iteration/500)
            path = os.path.join(cfg.output_dir, run_id, 'test', str(folder_id))

    else:
        path = os.path.join(cfg.output_dir, run_id, 'val', str(ep))
        
    if not os.path.exists(path):
        os.makedirs(path)

    input_img_path = '%s/%05d_input.jpg' % (path, iteration)
    mask_path = '%s/%05d_mask.jpg' % (path, iteration)
    output_path = '%s/%05d_output.jpg' % (path, iteration)
    
    if cfg.partial_map:
        partial_dec_out_path = '%s/%05d_pd_output.jpg' % (path, iteration)
    if cfg.heatmap_prediction:
        gt_hmap_path = '%s/%05d_gt_hmap.jpg' % (path, iteration)
        p_hmap_path = '%s/%05d_p_hmap.jpg' % (path, iteration)


    n_row = 1 if test else 4
    
    torchvision.utils.save_image(visuals['input'], input_img_path, normalize=True, nrow=n_row, range=(0, 1))
    torchvision.utils.save_image(visuals['mask'], mask_path, normalize=True, nrow=n_row, range=(0, 1.0))
    torchvision.utils.save_image(visuals['output'], output_path, normalize=True, nrow=n_row, range=(0, 1))
    if caranet:
        torchvision.utils.save_image(visuals['partial_d'], partial_dec_out_path, normalize=True, nrow=n_row, range=(0, 1))
    if cfg.heatmap_prediction:
        torchvision.utils.save_image(visuals['gt_hmap'], gt_hmap_path, normalize=True, nrow=n_row, range=(0, 1))
        torchvision.utils.save_image(visuals['p_hmap'], p_hmap_path, normalize=True, nrow=n_row, range=(0, 1))

    if test:
        # print("mask: ",torch.unique(visuals['mask']))
        # print("output: ",torch.unique(visuals['output']))
        contour_path = '%s/%05d_contour.jpg' % (path, iteration)

        img = (visuals['input'].cpu().data.numpy()) * 255.0
        mask = (visuals['mask'].cpu().data.numpy()) * 255.0
        output = (visuals['output'].cpu().data.numpy()) * 255.0

        #print(img.shape, mask.shape, output.shape)

        # img = np.transpose(img,(2, 3, 1)).squeeze()
        # mask = np.transpose(mask,(2, 3, 1)).squeeze()
        # output = np.transpose(output,(2, 3, 1)).squeeze()

        img = img.squeeze()
        mask = mask.squeeze()
        output = output.squeeze()


        # print(img.shape, mask.shape, output.shape)
        mask = mask.astype(np.uint8)
        img = img.astype(np.uint8)
        # print("mask img: ",np.unique(mask), np.unique(img))
        contoured_img = draw_contour(img, mask)
        # print("contoured img: ",np.unique(contoured_img))

        # print("contour image shape: ",contoured_img.shape)
        output = output.astype(np.uint8)
        # print("output: ",np.unique(output))
        contoured_img = draw_contour(contoured_img, output, color=(255, 0, 0))
        cv2.imwrite(contour_path, contoured_img)