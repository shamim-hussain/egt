import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np
import os

from lib.base.dotdict import HDict
from lib.data.datasets.zinc_full import SVDDataset
from lib.models.zinc_full.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme
from lib.training.schemes.zinc._eval import ZINCEval


class ZincFullDCSVD(ZINCEval, BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name       = 'zinc_full',
            dataset_path       = "datasets/ZINC_full/ZINC_full.h5",
            num_virtual_nodes  = 0,
            rlr_monitor        = 'val_mae',
            save_best_monitor  = 'val_mae',
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, SVDDataset
    
    def get_model_config(self):
        config = self.config
        
        model_config, _ = super().get_model_config()
        model_config.update(
            readout_edges = False,
            num_virtual_nodes = config.num_virtual_nodes,
        )
        return model_config, DCSVDTransformer
    
    def get_loss(self):
        loss = losses.MeanAbsoluteError(name='MAE')
        return loss
    
    def get_metrics(self):
        return ['mae']



SCHEME = ZincFullDCSVD




