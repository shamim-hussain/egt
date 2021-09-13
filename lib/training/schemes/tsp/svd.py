import tensorflow as tf
from tensorflow.keras import (optimizers, losses, metrics)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

from lib.base.dotdict import HDict
from lib.data.datasets.tsp import SVDDataset
from lib.models.tsp.dc import DCSVDTransformer
from lib.training.schemes.scheme_base import BaseSVDModelScheme


class TSPDCSVD(BaseSVDModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name      = 'tsp',
            batch_size        = 8,
            prediction_bmult  = 3,
            include_xpose     = True,
            save_best_monitor = 'val_xent',
            rlr_monitor       = 'val_xent',
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, SVDDataset
    
    def get_model_config(self):
        config = self.config
        model_config, _ = super().get_model_config()
        model_config.update(
            use_node_embeddings = (config.edge_channel_type not in
                                   ['residual','constrained']) ,
        )
        return model_config, DCSVDTransformer
    
    def get_loss(self):
        loss = losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                    name='xentropy')
        return loss

    def get_metrics(self):
        xent = metrics.SparseCategoricalCrossentropy(from_logits=True,
                                                     name='xent')
        return [xent,'acc']
    
    def do_evaluations_on_split(self,split):
        dataset = getattr(self,split)
        model = self.model
        strategy = self.strategy
        
        targs = []
        preds = []
        prog_bar = tqdm()
        def collate_fn(fmat,tmat,outp):
            bool_mask = (fmat.numpy().squeeze() >= 0)
            targ = tmat.numpy().squeeze()[bool_mask]
            pred = outp.numpy().squeeze().argmax(-1)[bool_mask]
            
            targs.append(targ)
            preds.append(pred)
            prog_bar.update()
        
        
        @tf.function
        def prediction_step(*inputs):
            return model(inputs, training=False)
        
        if self.config.distributed:
            dataset = strategy.experimental_distribute_dataset(dataset)
        
        @tf.function
        def make_predictions():
            for i,t in dataset:
                inps = tuple(i[n] for n in self.model.input_names)
                fmat = i['feature_matrix']
                tmat = t['target']
                
                if not self.config.distributed:
                    outp = prediction_step(inps)
                else:
                    outp = strategy.experimental_run_v2(prediction_step, args=inps)
                    outp = tf.concat(outp.values, axis=0)
                    fmat = tf.concat(fmat.values, axis=0)
                    tmat = tf.concat(tmat.values, axis=0)
                
                tf.py_function(collate_fn, [fmat, tmat, outp], [])
            
        make_predictions()

        targs = np.concatenate(targs, axis=0)
        preds = np.concatenate(preds, axis=0)
        prog_bar.close()

        acc = accuracy_score(targs, preds)
        prec = precision_score(targs, preds)
        rec = recall_score(targs, preds)
        f1 = f1_score(targs,preds)

        print(f'Accuracy = {acc}')
        print(f'Precision = {prec}')
        print(f'Recall = {rec}')
        print(f'f1 = {f1}')
        
        save_path = os.path.join(self.config.predictions_path,f'{split}_evals.txt')
        with open(save_path, 'a') as fl:
            print(f'Accuracy = {acc}', file=fl)
            print(f'Precision = {prec}', file=fl)
            print(f'Recall = {rec}', file=fl)
            print(f'f1 = {f1}', file=fl)


SCHEME = TSPDCSVD




