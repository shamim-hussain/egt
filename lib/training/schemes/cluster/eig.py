import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np

from lib.base.dotdict import HDict
from lib.data.datasets.sbm_cluster import EigenDataset
from lib.models.sbm_cluster.dc import DCEigTransformer
from lib.training.schemes.scheme_base import BaseEigModelScheme
from lib.training.schemes.cluster._eval import SBMCLUSTEREval


class SBMCDCEig(SBMCLUSTEREval, BaseEigModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name  = 'sbm_cluster',
            class_sizes   = [19695, 19222, 19559, 19417, 19801, 20139,],
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, EigenDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, DCEigTransformer
    
    def get_loss(self):
        class_sizes = np.array(self.config.class_sizes, dtype='float32')
        class_weights = class_sizes.sum() - class_sizes
        class_weights = class_weights/class_weights.sum()
        class_weights = tf.constant(class_weights, dtype=tf.float32)
        def loss(y_true, y_pred):
            weights = tf.gather(class_weights, tf.cast(y_true, tf.int32))
            w_xent = weights * losses.sparse_categorical_crossentropy(y_true, y_pred, 
                                                                from_logits=True, axis=-1)
            return w_xent
        
        return loss
    
    def get_metrics(self):
        acc = metrics.SparseCategoricalAccuracy(name='acc')
        return [acc]
    


SCHEME = SBMCDCEig




