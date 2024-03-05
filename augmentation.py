#system library
import sys
sys.path.append('../')


#python library
import random

#torch library
import torch

#third-party library
import albumentations as A
import cv2


#other project files
import config as cfg 


class Augmentation():

    def __init__(self, args):
        super(Augmentation, self).__init__()
        self.sz = args.sz
        self.transform_structure = self._get_geometric_transformation()
        self.transform_pixel = self._get_pixel_level_transformation()

    def  _get_geometric_transformation(self):
        rotation_limit = random.randint(5,10)
        transform = A.Compose(
            [
                A.CenterCrop (height = self.sz, width = self.sz, p=1.0),
                A.Rotate(limit=rotation_limit, p=0.5),
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
            ],
            additional_targets = {'image0':'image', 'image1':'image'}
        )
        return transform

    def _get_pixel_level_transformation(self):
        grid_size = random.randrange(3, 9, 2)
        noise_mean = random.randint(-10, 10)
        window = random.randrange(3, 11, 2)
        noise_variance = (5.0, 15.0)
        transform = A.Compose(
            [
                #A.CLAHE(clip_limit=4.0, tile_grid_size=(grid_size, grid_size), p=0.5),
                #A.GaussNoise(var_limit=noise_variance, mean=noise_mean, p=0.5),
                #A.GaussianBlur(blur_limit=(window, window), p=0.5),
                A.RandomBrightnessContrast(p=0.5),    
                A.RandomGamma(p=0.5)
            ]
        )
        return transform

    def generation(self, img, target):
        # scale_factor = params.scale_factor
        # if isReduced:
        #     width = int(img.shape[1] * scale_factor)
        #     height = int(img.shape[0] * scale_factor)
        #     dsize = (width, height)
        #     img = cv2.resize(img, dsize)
        #     target = cv2.resize(target, dsize)
        transformed = self.transform_structure(image=img, image0=target)
        aug_img, aug_label = transformed['image'], transformed['image0']
        #transformed_pixel = self.transform_pixel(image=aug_img)
        #aug_img = transformed_pixel['image']
        return aug_img, aug_label

