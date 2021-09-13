import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np
import os

from lib.base.dotdict import HDict
from lib.data.datasets.pcqm4m import CachedSVDMatrixDataset, max_node_feat_dims, max_edge_feat_dims
from lib.models.pcqm4m.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme
from lib.training.schemes.pcqm4m._eval import PCQM4MEval


class PCQM4MDCSVD(PCQM4MEval, BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name        = 'pcqm4m',
            cache_dir           = HDict.L('c:f"./data/{c.dataset_name.upper()}/mat"'),
            num_virtual_nodes   = 0,
            cache_data          = True,
            max_shuffle_len     = 100000,
            random_neg          = False,
            # svd_cache_data      = True,
            svd_cache_path      = HDict.L('c:f"./data/{c.dataset_name.upper()}/mat/svd"'),
            num_svd_features    = 64,
            t_num_features      = 8,
            combined_cache_path = HDict.L('c:f"./data/{c.dataset_name.upper()}/mat/combined_{c.t_num_features}"'),
            sel_svd_features    = 8,
            max_feats           = False, 
            weight_file         = "",
            save_best_monitor   = 'val_mae',
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        config = self.config
        
        dataset_config, _ = super().get_dataset_config()
        dataset_config.update(
            cache_data          = config.cache_data         ,
            cache_path          = config.cache_dir          ,
            svd_cache_path      = config.svd_cache_path     ,
            t_num_features      = config.t_num_features     ,
            combined_cache_path = config.combined_cache_path,
        )
        return dataset_config, CachedSVDMatrixDataset
    
    def get_model_config(self):
        config = self.config
        
        model_config, _ = super().get_model_config()
        model_config.update(
            num_svd_features  = config.t_num_features    ,
            readout_edges     = False,
            num_virtual_nodes = config.num_virtual_nodes,
        )
        if self.config.max_feats:
            model_config.update(
                node_feat_dims = tuple(max_node_feat_dims),
                edge_feat_dims = tuple(max_edge_feat_dims),
            )
        return model_config, DCSVDTransformer
    
    def get_loss(self):
        loss = losses.MeanAbsoluteError(name='MAE')
        return loss
    
    def get_metrics(self):
        return ['mae']



SCHEME = PCQM4MDCSVD




