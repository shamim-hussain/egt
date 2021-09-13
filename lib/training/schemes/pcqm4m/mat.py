import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np
import os

from lib.base.dotdict import HDict
from lib.data.datasets.pcqm4m import MatrixDataset, max_node_feat_dims, max_edge_feat_dims
from lib.models.pcqm4m.dc import DCTransformer
from lib.training.schemes.scheme_base import BaseAdjModelScheme
from lib.training.schemes.pcqm4m._eval import PCQM4MEval

class PCQM4MDCMAT(PCQM4MEval, BaseAdjModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name       = 'pcqm4m',
            num_virtual_nodes  = 0,
            cache_data         = True,
            max_shuffle_len    = 100000,
            max_feats          = False,
            save_best_monitor  = 'val_mae',
            weight_file        = "",
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        config = self.config
        
        dataset_config, _ = super().get_dataset_config()
        dataset_config.update(
            cache_data = config.cache_data,
            cache_path = config.cache_dir,
        )
        return dataset_config, MatrixDataset
    
    def get_model_config(self):
        config = self.config
        
        model_config, _ = super().get_model_config()
        model_config.update(
            readout_edges     = False,
            num_virtual_nodes = config.num_virtual_nodes,
        )
        if self.config.max_feats:
            model_config.update(
                node_feat_dims = tuple(max_node_feat_dims),
                edge_feat_dims = tuple(max_edge_feat_dims),
            )
        return model_config, DCTransformer
    
    def get_loss(self):
        loss = losses.MeanAbsoluteError(name='MAE')
        return loss
    
    def get_metrics(self):
        return ['mae']


SCHEME = PCQM4MDCMAT




