import numpy as np
import cv2 

import torch 
from torchvision.transforms import Compose
import torchvision 
from torch.utils.data.dataloader import DataLoader as DL

import rospy 
from nav_msgs.msg import GridCells
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import configs.models
import configs.base
from models.build import build_model

from datetime import datetime

import os 
import csv
import sys

class PredictPatch:

    def __init__(self, patch_shape, stride, waterfall_shape, model_path, config_path, encoder, decoder):       
        
        self.cmap = {0: (255, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
        
        self.load_model(model_path, config_path, encoder, decoder) 
        self.patch_shape = patch_shape
        self.stride = stride 

        # warmup gpu by passing empty tensors 
        for _ in range(10):
            warmup = torch.zeros(100).to(device="cuda")

        print("Loading model to device...")
        self.model.to(torch.device('cuda'))
        print("Loaded model to device!")

    def mask2rgb(self, idx_img):
        """Converts the class indices to rgb using the cmap

        Args:
            idx_img (_type_): class index image 

        Returns:
            _type_: rgb image 
        """

        rgb_img = np.empty(idx_img.shape+(3,), dtype='uint8')
        vs = []
        for k,v in self.cmap.items():
            rgb_img[idx_img==k] = v
        return rgb_img

    def onehot2rgb(self, hot_img):
        """Converts image from onehot encoding to rgb

        Args:
            hot_img (_type_): one hot encoding image

        Returns:
            _type_: rgb image
        """

        idx_img = np.argmax(hot_img, axis=1)
        confidence = np.max(hot_img, axis=1)
        rgb_img = self.mask2rgb(idx_img)
        return idx_img, rgb_img, confidence

    def load_model(self, model_path, config_file, encoder, decoder):
        """Loads model from disk 

        Args:
            model_path (_type_): model file path 
            config_file (_type_): configuration file path 
            encoder (_type_): encoder type
            decoder (_type_): decoder type 
        """
        
        encoder_choices = ('mpvit', 'sima_tiny','sima_mini')
        encoder_configs = {k: configs.models.__getattribute__(k) for k in encoder_choices}

        decoder_choices = ('mlp', 'conv', 'atrous')
        decoder_configs = {k: configs.models.__getattribute__(k) for k in decoder_choices}

        config = configs.base._C.clone()                            # Base Configurations
        config.merge_from_list(encoder_configs[encoder]())     # Architecture defaults
        config.merge_from_list(decoder_configs[decoder]())
        if config_file:
            config.merge_from_file(config_file)                # User Customizations
        config.freeze()
    
        self.model = build_model(config)

        state_dict = torch.load(model_path, map_location="cpu")
        msg = self.model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights loaded with msg: {}'.format(msg))

        self.model.eval()

        self.data_transforms = Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(config.DATA.MEAN, config.DATA.STD)
            ])

        print('Model built') 

    def sliding_window_positions(self, image):
        """Calculate patch positions inside waterfall 

        Args:
            image (_type_): waterfall 

        Returns:
            _type_: image shape and patch row and column indices 
        """
        h, w = image.shape[:2]
        row_idxs = [r for r in range(0, h - self.patch_shape[0] + 1, self.stride[0])]
        if h - row_idxs[-1] != self.patch_shape[0]:
            row_idxs.append(h - self.patch_shape[0])
        col_idxs = [c for c in range(
            0, w - self.patch_shape[1] + 1, self.stride[1])]
        if w - col_idxs[-1] != self.patch_shape[1]:
            col_idxs.append(w - self.patch_shape[1])
        return h, w, row_idxs, col_idxs
    
    def image_to_tensor(self, image, h, w, row_idxs, col_idxs):
        """Extract patches from image and place into tensor 

        Args:
            image (_type_): input image 
            h (_type_): image height
            w (_type_): image width 
            row_idxs (_type_): patch row indices
            col_idxs (_type_): patch column indices 

        Returns:
            _type_: closest image positions for each patch created 

        Yields:
            _type_: tensor made of patches along first row of sliding window 
        """

        # create image positions 
        xs, ys = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
        indices = np.ones(image.shape) * -1.0
        distances = np.ones(image.shape) * sys.float_info.max

        for i, r in enumerate(row_idxs):
            tensor = torch.empty(len(col_idxs), 1, *self.patch_shape)
            for j, c in enumerate(col_idxs):
                # extract patch and convert to tensor 
                patch = image[r:r + self.patch_shape[0], c:c + self.patch_shape[1]]
                patch_added_dim = np.expand_dims(patch, axis=2)
                tensor[j, 0, :, :] = self.data_transforms(patch_added_dim)

                # calculate image positions closest to current patch and store them  
                centre = (r + self.patch_shape[0] / 2, c + self.patch_shape[1] / 2)
                new_distances = (xs[r:r + self.patch_shape[0], c:c + self.patch_shape[1]] - centre[0]) \
                    + (ys[r:r + self.patch_shape[0], c:c + self.patch_shape[1]] - centre[1])
                closer = distances[r:r + self.patch_shape[0], c:c + self.patch_shape[1]] > new_distances
                distances[r:r + self.patch_shape[0], c:c + self.patch_shape[1]][closer] = new_distances[closer]
                indices[r:r + self.patch_shape[0], c:c + self.patch_shape[1]][closer] = i * len(col_idxs) + j

            # yield current tensor 
            yield tensor
        
        # return patches' closest image positions 
        return indices
    

    def predict(self, image, device='cuda'):
        """Extract patches to model, predict patches and recompose output from patch outputs 

        Args:
            image (_type_): input image 
            device (str, optional): device type (cuda or cpu). Defaults to 'cuda'.

        Returns:
            _type_: output class image, rgb image, confidence image  
        """

        row_rgbs = []
        row_confidences = []
        row_classes = []

        # calculate sliding window positions for patch extraction
        h, w, row_idxs, col_idxs = self.sliding_window_positions(image)
        
        with torch.no_grad():
            
            # patch generator
            patches_gen = self.image_to_tensor(image, h, w, row_idxs, col_idxs)
            patch_indices = None

            while True:
                try:
                    tensor = next(patches_gen)
                    
                    # perform model inference on patch 
                    tensor = tensor.to(device)
                    output = torch.softmax(self.model(tensor).to("cpu"), dim=1).numpy()
                    #output = self.softmax_with_temperature(self.model(tensor).to("cpu").numpy())
                    classes, rgbs, confidences = self.onehot2rgb(output)

                    row_rgbs.append(rgbs.astype(np.uint8))
                    row_confidences.append(confidences)
                    row_classes.append(classes)

                except StopIteration as e:
                    # generator function final output 
                    patch_indices = e.value
                    break

        # reconstruct patch outputs to single image 
        all_classes, all_rgbs, all_confidences = self.patches_to_image(row_classes, row_rgbs, row_confidences, (h, w), patch_indices, row_idxs, col_idxs)
        return all_classes, all_rgbs, all_confidences
    

    def patches_to_image(self, patch_classes, patch_rgbs, patch_confidences, output_image_shape, indices, row_idxs, col_idxs):  
        """Reconstruct class image, rgb image and confidence image from patches 

        Args:
            patch_classes (_type_): class patch images 
            patch_rgbs (_type_): rgb patch images 
            patch_confidences (_type_): confidence patch images 
            output_image_shape (_type_): output image height and width 
            indices (_type_): patch indices inside original image 
            row_idxs (_type_): patch row indices
            col_idxs (_type_): patch column indices 

        Returns:
            _type_: output class image, rgb image, confidence image
        """
        combined_rgbs = np.zeros((*output_image_shape,3), dtype=np.uint8)
        combined_confidences = np.zeros(output_image_shape)
        combined_classes = np.ones(output_image_shape) * -1.0

        for i, (classes, rgb, confidence) in enumerate(zip(patch_classes, patch_rgbs, patch_confidences)):
            for j in range(rgb.shape[0]):
                idx = i * len(col_idxs) + j
                origin = (row_idxs[i], col_idxs[j])
                
                # for each patch get corresponding image positions 
                to_change = indices == idx
                y_idxs, x_idxs = np.nonzero(to_change)
                shifted_y_idxs = y_idxs - origin[0]
                shifted_x_idxs = x_idxs - origin[1]

                # place patch output in the correct position inside the image 
                combined_rgbs[y_idxs, x_idxs, :] = rgb[j, shifted_y_idxs, shifted_x_idxs, :]
                combined_confidences[y_idxs, x_idxs] = confidence[j, shifted_y_idxs, shifted_x_idxs]
                combined_classes[y_idxs, x_idxs] = classes[j, shifted_y_idxs, shifted_x_idxs]

        return combined_classes, combined_rgbs, combined_confidences
    
    @staticmethod
    def prediction_to_hsv_encoding(image, class_outputs, confidences):
        """Convert model output to HSV encoding 

        Args:
            image (_type_): input image 
            class_outputs (_type_): rgb image output
            confidences (_type_): confidence image output 

        Returns:
            _type_: hsv encoding as rgb and hsv image 
        """
        
        classes_hsv = cv2.cvtColor(class_outputs, cv2.COLOR_RGB2HSV)
        classes_hue = cv2.split(classes_hsv)[0]
        
        # assign rgb output to hue, image intensities to value, and confidence to saturation
        # when confidence very low, image goes grayscale, when high, it shows more intensely the rgb colour
        h = classes_hue
        v = image 
        s = (confidences * 255.0).astype(np.uint8)

        combined = cv2.merge([h, s, v])
        output = cv2.cvtColor(combined, cv2.COLOR_HSV2RGB)

        return output, combined
    