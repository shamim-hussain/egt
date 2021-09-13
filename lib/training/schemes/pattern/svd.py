import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, log_loss
import numpy as np
import os

from lib.base.dotdict import HDict
from lib.data.datasets.sbm_pattern import SVDDataset
from lib.models.sbm_pattern.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme
from lib.base.genutil.losses import WeightedSparseXEntropyLoss, WeightedSparseXEntropyMetric
from lib.training.schemes.pattern._eval import SBMPATTERNEval

class SBMPDCSVD(SBMPATTERNEval, BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name  = 'sbm_pattern',
            class_sizes   = [979220, 209900],
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
    


SCHEME = SBMPDCSVD




