import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np


from lib.base.dotdict import HDict
from lib.data.datasets.sbm_cluster import SVDDataset
from lib.models.sbm_cluster.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme
from lib.base.genutil.losses import WeightedSparseXEntropyLoss, WeightedSparseXEntropyMetric
from lib.training.schemes.cluster._eval import SBMCLUSTEREval

class SBMCDCSVD(SBMCLUSTEREval, BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name  = 'sbm_cluster',
            class_sizes   = [19695, 19222, 19559, 19417, 19801, 20139,],
            rlr_monitor        = 'val_xent',
            save_best_monitor  = 'val_xent',
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, SVDDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, DCSVDTransformer
    
    def get_loss(self):
        wxent = WeightedSparseXEntropyLoss(class_weights=None, 
                                           class_sizes=self.config.class_sizes,
                                           from_logits=True,
                                           name='xentropy')
        return [wxent]
    
    def get_metrics(self):
        wxent = WeightedSparseXEntropyMetric(class_weights=None,
                                             class_sizes=self.config.class_sizes,
                                             from_logits=True,
                                             name='xent')
        acc = metrics.SparseCategoricalAccuracy(name='acc')
        return [wxent, acc]
    
    


SCHEME = SBMCDCSVD




