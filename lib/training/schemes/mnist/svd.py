import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
import numpy as np
import os

from lib.base.dotdict import HDict
from lib.data.datasets.mnist import SVDDataset
from lib.models.mnist.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme
# from lib.training.schemes.evaluation import save_results


class MNISTDCSVD(BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name       = 'mnist',
            save_best_monitor  = 'val_xent',
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
        )
        return model_config, DCSVDTransformer
    
    def get_loss(self):
        loss = losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                    name='xentropy')
        return loss

    def get_metrics(self):
        xent = metrics.SparseCategoricalCrossentropy(from_logits=True, name='xent')
        return ['acc',xent]
    
    def do_evaluations_on_split(self,split):
        loss, acc, xent, *_ = self.model.evaluate(getattr(self,split))
        print(f'{split} accuracy = {acc:0.5%}')
        print(f'{split} crossentropy = {xent:0.6f}')
        
        save_path = os.path.join(self.config.predictions_path,f'{split}_evals.txt')
        with open(save_path, 'a') as fl:
            print(f'{split} accuracy = {acc:0.5%}', file=fl)
            print(f'{split} crossentropy = {xent:0.6f}', file=fl)
            
        # save_results(
        #     dataset_name = self.config.dataset_name,
        #     model_name   = self.config.model_name,
        #     split        = split,
        #     metrics      = dict(
        #         accuracy     = acc.tolist(),
        #         crossentropy = xent.tolist(),
        #         loss         = loss.tolist(),
        #     ),
        #     configs      =self.config,
        #     state        =self.state,
        #     parent_dir   =self.config.predictions_path,
        # )
        

SCHEME = MNISTDCSVD





