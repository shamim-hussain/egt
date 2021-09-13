
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
# from lib.training.schemes.evaluation import save_results


class SBMCLUSTEREval:
    def do_evaluations_on_split(self,split):
        def accuracy_SBM(targets, preds):
            S = targets
            C = preds
            
            CM = confusion_matrix(S,C).astype(np.float32)
            nb_classes = CM.shape[0]
            
            pr_classes = np.zeros(nb_classes)
            for r in range(nb_classes):
                cluster = np.where(targets==r)[0]
                if cluster.shape[0] != 0:
                    pr_classes[r] = CM[r,r]/ float(cluster.shape[0])
                else:
                    pr_classes[r] = 0.0
            acc = np.sum(pr_classes)/ float(nb_classes)
            return acc
        
        dataset = getattr(self,split)
        model = self.model
        strategy = self.strategy
        
        targs = []
        preds = []
        prog_bar = tqdm()
        def collate_fn(nodef,trgs,outp):
            bool_mask = (nodef.numpy().ravel() >= 0)
            targ = trgs.numpy().ravel()[bool_mask]
            pred=outp.numpy().argmax(-1).ravel()[bool_mask]
            
            targs.append(targ)
            preds.append(pred)
            prog_bar.update()
        
        @tf.function
        def prediction_step(*inputs):
            return tf.nn.softmax(model(inputs, training=False))
        
        if self.config.distributed:
            dataset = strategy.experimental_distribute_dataset(dataset)
        
        @tf.function
        def make_predictions():
            for i,t in dataset:
                inps = tuple(i[n] for n in self.model.input_names)
                nodef = i['node_features']
                trgs = t['target']

                if not self.config.distributed:
                    outp = prediction_step(inps)
                else:
                    outp = strategy.experimental_run_v2(prediction_step, args=inps)
                    outp = tf.concat(outp.values, axis=0)
                    nodef = tf.concat(nodef.values, axis=0)
                    trgs = tf.concat(trgs.values, axis=0)
                
                tf.py_function(collate_fn, [nodef, trgs, outp], [])
        
        make_predictions()

        targs = np.concatenate(targs, axis=0)
        preds = np.concatenate(preds, axis=0)
        prog_bar.close()
        
        classes = (np.eye(targs.max()+1)[targs]).sum(0)
        
        micro_rec = recall_score(targs, preds, average='micro')
        macro_rec = recall_score(targs, preds, average='macro')
        acc = accuracy_score(targs, preds)
        wacc = accuracy_SBM(targs, preds)

        print(f'Accuracy = {acc:0.5%}')
        print(f'Micro Recall = {micro_rec:0.5%}')
        print(f'Macro Recall = {macro_rec:0.5%}')
        print(f'Weighted Accuracy = {wacc:0.5%}')
        print(f'Binned classes:{classes}')
        
        save_path = os.path.join(self.config.predictions_path,f'{split}_evals.txt')
        with open(save_path, 'a') as fl:
            print(f'Accuracy = {acc:0.5%}', file=fl)
            print(f'Micro Recall = {micro_rec:0.5%}', file=fl)
            print(f'Macro Recall = {macro_rec:0.5%}', file=fl)
            print(f'Weighted Accuracy = {wacc:0.5%}', file=fl)
            print(f'Binned classes:{classes}')
        
        # save_results(
        #     dataset_name = self.config.dataset_name,
        #     model_name   = self.config.model_name,
        #     split        = split,
        #     metrics      = dict(
        #         accuracy          = acc,
        #         micro_recall      = micro_rec,
        #         macro_recall      = macro_rec,
        #         weighted_accuracy = wacc,
        #         binned_classes    = classes.tolist(),
        #     ),
        #     configs      =self.config,
        #     state        =self.state,
        #     parent_dir   =self.config.predictions_path,
        # )