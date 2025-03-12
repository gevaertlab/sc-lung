import importlib
import random

import cv2
import numpy as np
import os
import datetime
import pandas as pd

from dataset import get_dataset

def get_pretrained(model_name = "pannuke"):
    assert model_name in ['resnet', 'pannuke', 'monusac']
    pretrain_dir = "./sample_data"
    if model_name == "pannuke":
        model = os.path.join(pretrain_dir, "hovernet_fast_pannuke_type_custom.tar")
    elif model_name == "resnet":
        model = os.path.join(pretrain_dir, "ImageNet-ResNet50-Preact_pytorch.tar")
    elif model_name == "monusac":
        model = os.path.join(pretrain_dir, "hovernet_fast_monusac_type_custom.tar")
    return model

class Config(object):
    """Configuration file."""

    def __init__(self):
        self.seed = 10

        self.source_path = "."
        self.data_path = os.path.join(self.source_path, "output")
        self.save_path = os.path.join(self.source_path, "output", "hovernet_models")

        self.dataset_class = "orion_GMM_7_class" # corresponds to the dataset class defined in dataset.py

        nr_type = 7 # number of nuclear types (need to include the background)

        train_samples = ['CRC08']
        val_samples = ['CRC08']
        
        # Do add the weight for the background at the first position !!!
        class_weights = [1.00, 0.92, 0.74, 0.87, 0.73, 2.09, 1.60] # Do add the weight for the undefined class at the first position !!!
        assert len(class_weights) == nr_type
        
        epoch = 12
        self.debug = 0

        # None to start from scratch
        pretrained = get_pretrained(model_name="pannuke")
        
        self.logging = True
    
        # turn on debug flag to trace some parallel processing problems more easily
        # self.debug = True

        model_name = "hovernet"
        model_mode = "fast" # choose either `original` or `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        # whether to predict the nuclear type, availability depending on dataset!
        self.type_classification = True

        # shape information - 
        # below config is for original mode. 
        # If original model mode is used, use [270,270] and [80,80] for act_shape and out_shape respectively
        # If fast model mode is used, use [256,256] and [164,164] for act_shape and out_shape respectively
        aug_shape = [540, 540] # patch shape used during augmentation (larger patch may have less border artefacts)
        act_shape = [256, 256] # patch shape used as input to network - central crop performed after augmentation
        out_shape = [164, 164] # patch shape at output of network

        if model_mode == "original":
            if act_shape != [270,270] or out_shape != [80,80]:
                raise Exception("If using `original` mode, input shape must be [270,270] and output shape must be [80,80]")
        if model_mode == "fast":
            if act_shape != [256,256] or out_shape != [164,164]:
                raise Exception("If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]")

        self.log_dir = os.path.join(self.save_path, "logs/") # where checkpoints will be saved

        # paths to training and validation patches
        self.train_dir_list = [
            os.path.join(self.data_path, "training_data", f"{sample_id}/540x540_164x164") for sample_id in train_samples
        ]
        self.valid_dir_list = [
            os.path.join(self.data_path, "training_data", f"{sample_id}/540x540_164x164") for sample_id in val_samples
        ]
        
        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * parsing config to the running state and set up associated variables
        self.dataset = get_dataset(self.dataset_class)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode, pretrained, epoch, class_weights)
