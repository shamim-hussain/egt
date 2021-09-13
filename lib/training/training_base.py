import tensorflow as tf
import numpy as np
from tensorflow.keras import (optimizers, losses, metrics, callbacks)


import json
import os

from lib.base.dotdict import HDict
HDict.L.update_globals({'path':os.path})

from lib.data.reader import ExcludeFeatures, CreateTargets
from lib.base.callbacks.checkpoint import CheckpointCallback, SaveWhenCallback

def read_config_from_file(config_file):
    with open(config_file, 'r') as fp:
        return json.load(fp)

def save_config_to_file(config, config_file):
    with open(config_file, 'w') as fp:
        return json.dump(config, fp, indent='\t')

class TrainingBase:
    def __init__(self, config=None):
        self.config_input = config
        self.config = self.get_default_config()
        if config is not None:
            for k in config.keys():
                if not k in self.config:
                    raise KeyError(f'Unknown config "{k}"')
            self.config.update(config)
        
        self.state = self.get_default_state()
        
        self.pred_flag = False
        self.eval_flag = False
    
    def get_dataset_config(self):
        return {}, None
    
    def get_dataset(self, splits=['training','validation']):
        dataset_config, dataset_class = self.get_dataset_config()
        if dataset_class is None:
            raise NotImplementedError
        return dataset_class(**dataset_config, splits = splits)
    
    def get_excluded_features(self):
        return []
    
    def get_model_config(self):
        return {}, None
    
    def get_model(self):
        model_config, model_class = self.get_model_config()
        if model_class is None:
            raise NotImplementedError
        return model_class(**model_config)
    
    def get_optimizer(self):
        config = self.config
        
        optim_dict = dict(
            adam    = optimizers.Adam    ,
            rmsprop = optimizers.RMSprop ,
            sgd     = optimizers.SGD     ,
        )
        optim = optim_dict[config.optimizer]
        if config.gradient_clipval is None:
            return optim(learning_rate = config.initial_lr)
        else:
            return optim(learning_rate = config.initial_lr, 
                         clipvalue     = config.gradient_clipval)
    
    def get_loss(self):
        raise NotImplementedError
    
    def get_metrics(self):
        return None

    def get_default_config(self):
        return HDict(
            scheme            = None,
            model_name        = 'unnamed_model',
            distributed       = False,
            batch_size        = HDict.L('c:32 if c.distributed else 128'),
            initial_lr        = 5e-4,
            gradient_clipval  = None,
            num_epochs        = 1000,
            dataset_path      = 'datasets/gnn_benchmark.h5',
            save_path         = HDict.L('c:path.join("models",c.model_name)'),
            checkpoint_path   = HDict.L('c:path.join(c.save_path,"checkpoint")'),
            log_path          = HDict.L('c:path.join(c.save_path,"logs")'),
            config_path       = HDict.L('c:path.join(c.save_path,"config")'),
            summary_path      = HDict.L('c:path.join(c.save_path,"summary")'),
            saved_model_path  = HDict.L('c:path.join(c.save_path,"saved", c.model_name)'),
            rlr_factor        = 0.5, 
            rlr_patience      = 10,
            rlr_monitor       = HDict.L("c: c.save_best_monitor"),
            min_lr_factor     = 0.01,
            stopping_lr       = 0.,
            steps_per_epoch   = None,
            validation_steps  = None,
            save_best         = True,
            save_when         = HDict.L("c: '' if not c.save_best "+
                                        "else 'epoch;'+c.save_best_monitor+'<=save_best_value;epoch{epoch:0>4d}'"),
            save_best_monitor = 'val_loss',
            stopping_patience = 0,
            predictions_path  = HDict.L('c:path.join(c.save_path,"predictions")'),
            weight_file       = ':',
            prediction_bmult  = 2,
            optimizer         = 'adam',
        )
    
    def get_default_state(self):
        state =  HDict(
            current_epoch = tf.Variable(0, trainable=False, name="current_epoch"),
            global_step = tf.Variable(0, trainable=False, name="global_step"),
        )
        if self.config.save_best:
            state.update(
                save_best_value = tf.Variable(np.inf, trainable=False, 
                                              name="save_best_value"),
                save_best_epoch = tf.Variable(0, trainable=False, 
                                              name="save_best_epoch"),
        )
        if self.config.rlr_factor<1.0:
            state.update(
                last_reduce_lr = tf.Variable(0, trainable=False, 
                                              name="last_reduce_lr"),
        )
        return state
    
    def get_state_updates(self):
        config = self.config
        updates =  HDict(
            on_batch_end = [lambda model, state, *a, **kw: state.global_step.assign_add(1)],
            on_epoch_end = [lambda model, state, *a, **kw: state.current_epoch.assign_add(1)],                              
        )
        
        if config.save_best:
            monitor = config.save_best_monitor
            rlrp = config.rlr_patience
            rlrf = config.rlr_factor
            minlr = config.initial_lr * config.min_lr_factor
            stplr = config.stopping_lr
            def save_best_update(model, state, epoch, logs=None,
                                 *args, **kwargs):
                logs = logs or {}
                new_value = logs.get(monitor, np.inf)
                old_value = state.save_best_value.numpy()
            
                old_epoch = state.save_best_epoch.numpy()
                new_epoch = state.current_epoch.numpy()
                
                if new_value < old_value:
                    state.save_best_value.assign(new_value)
                    state.save_best_epoch.assign(new_epoch)
                    print(f'\nSAVE BEST: {monitor} improved from (epoch:{old_epoch},value:{old_value:0.5f})'+
                          f' to (epoch:{new_epoch},value:{new_value:0.5f})',flush=True)
                else:
                    print(f'\nSAVE BEST: {monitor} did NOT improve from'+
                          f' (epoch:{old_epoch},value:{old_value:0.5f})',flush=True)
                    
                    # RLR logic
                    if rlrf < 1.0:
                        last_reduce_lr = state.last_reduce_lr.numpy()
                        epoch_gap = (new_epoch - max(old_epoch, last_reduce_lr))
                        if epoch_gap >= rlrp:
                            model.optimizer.lr.assign(tf.maximum(model.optimizer.lr*rlrf, minlr))
                            state.last_reduce_lr.assign(new_epoch)
                            print(f'\nRLR: {monitor} did NOT improve for {epoch_gap} epochs,'+
                                  f' new lr = {model.optimizer.lr.numpy()}')
                
                # Stop training logic
                if model.optimizer.lr.numpy() < stplr:
                    model.stop_training = True
                    print(f'\nSTOP: lr fell below {stplr}, STOPPING TRAINING!')
                    
            updates.on_epoch_end.append(save_best_update)
        
        return updates
    
    def config_summary(self):
        for k,v in self.config.get_dict().items():
            print(f'{k} : {v}', flush=True)
    
    def save_config_file(self):
        os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
        save_config_to_file(self.config.get_dict(), self.config.config_path+'.json')
        save_config_to_file(self.config_input, self.config.config_path+'_input.json')
    
    def get_targets(self):
        return ['target']
    
    def get_batched_data(self):
        targets = self.get_targets()
        if len(targets)>0:
            map_fns = CreateTargets(targets)
        else:
            map_fns = None
            
        if self.eval_flag or self.pred_flag:
            return self.dataset.get_batched_data(self.config.batch_size*self.config.prediction_bmult,
                                                 map_fns=map_fns)
        else:
            return self.dataset.get_batched_data(self.config.batch_size,
                                                 map_fns=map_fns)
    
    def load_data(self, splits=['training','validation']):
        self.dataset = self.get_dataset(splits)
        if not self.config.cache_dir == '':
            self.dataset.cache(self.config.cache_dir)
        self.dataset.map(ExcludeFeatures(self.get_excluded_features()))

        self.trainset, *others = self.get_batched_data()
        self.valset, self.testset = None, None
        if others: self.valset, *others = others
        if others: self.testset, = others
    
    def model_summary(self):
        os.makedirs(os.path.dirname(self.config.summary_path), exist_ok=True)
        with open(self.config.summary_path+'.txt', 'w') as fp:
            print_fn = lambda *a,**kw: print(*a, **kw,file=fp)
            self.model.summary(print_fn=print_fn)
        
    
    def load_model(self):
        self.model_config = self.get_model()

        if self.config.distributed:
            self.strategy = tf.distribute.MirroredStrategy()
            self.strategy_scope = self.strategy.scope
        else:
            from contextlib import nullcontext
            self.strategy = None
            self.strategy_scope = nullcontext

        with self.strategy_scope():
            self.model = self.model_config.get_model()

            self.model_summary()

            opt = self.get_optimizer()
            loss = self.get_loss()
            metrics = self.get_metrics()
            
            self.model.compile(opt, loss, metrics)
    
    def load_state(self):
        os.makedirs(self.config.checkpoint_path, exist_ok=True)
        self.training_callbacks = self.get_callbacks()
        mchk_callback = CheckpointCallback(save_path = self.config.checkpoint_path,
                                           model     = self.model,
                                           state     = self.state,
                                           **self.get_state_updates())
        self.state_callbacks = [mchk_callback]
        self.callbacks = self.training_callbacks + self.state_callbacks
        
        with self.strategy_scope():
            mchk_callback.load_checkpoint()
        
        
    def get_base_callbacks(self):
        cbacks = []
        os.makedirs(self.config.log_path, exist_ok=True)
        logs_callback = callbacks.TensorBoard(log_dir=self.config.log_path)
        cbacks.append(logs_callback)
        
        if self.config.save_when:
            saved_model_dir = os.path.dirname(self.config.saved_model_path)
            os.makedirs(saved_model_dir, exist_ok=True)
            svwhn_callback = SaveWhenCallback(saved_model_dir,
                                              when              = self.config.save_when,
                                              state             = self.state,
                                              verbose           = 1,
                                              save_weights_only = True)
            cbacks.append(svwhn_callback)
        
        if self.config.stopping_patience > 0:
            estop_callback = callbacks.EarlyStopping(monitor  = 'val_loss',
                                                     verbose  = 1,
                                                     patience = self.config.stopping_patience)
            cbacks.append(estop_callback)
        
        return cbacks
    
    def get_callbacks(self):
        return self.get_base_callbacks()
    
    def get_additional_training_configs(self):
        return {}
    
    def train_model(self):
        self.model.fit(self.trainset, 
                       epochs            = self.config.num_epochs, 
                       validation_data   = self.valset,
                       callbacks         = self.callbacks,
                       initial_epoch     = self.state.current_epoch.numpy(),
                       steps_per_epoch   = self.config.steps_per_epoch,
                       validation_steps  = self.config.validation_steps,
                       **self.get_additional_training_configs()
                       )
    
    
    def execute_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_data()
        self.load_model()
        self.load_state()
        self.train_model()
        self.finalize_training(skip_init=True)
    
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.config.saved_model_path), exist_ok=True)
        save_path = self.config.saved_model_path+'.h5'
        self.model.save_weights(save_path)
        print(f'Saved model to {save_path}')
    
    def finalize_training(self, skip_init=False):
        if not skip_init:
            self.config_summary()
            self.load_model()
            self.load_state()
        self.save_model()
        print('DONE!!!')
    
    
    def get_latest_save_file(self):
        import re
        pattern = re.compile(r'(?<=epoch)[0-9]+')
        from pathlib import Path

        cur_epoch, cur_file = 0, ''
        for fp in Path(self.config.saved_model_path).parent.glob('*.h5'):
            m = pattern.search(fp.name)
            
            e = 0 if m is None else int(m.group())
            if e > cur_epoch:
                cur_epoch = e
                cur_file = str(fp)
        
        self.config.weight_file = cur_file
    
    
    def prepare_for_test(self):
        self.config_summary()
        self.load_data(splits=['training', 'validation', 'test'])
        self.load_model()
        
        if self.config.weight_file == ':':
            self.get_latest_save_file()
        
        if self.config.weight_file == '':
            self.config.weight_file = self.config.saved_model_path+'.h5'
            
        if self.config.weight_file == '-':
            self.load_state()
            print('LOADED TRAINING STATE FOR PREDICTIONS!')
        else:
            self.model.load_weights(self.config.weight_file, by_name=True)
            print(f'LOADED WEIGHT FILE "{self.config.weight_file}" FOR PREDICTIONS!')
    
    
    def make_predictions_on_split(self, split):
        raise NotImplementedError
    
    def do_evaluations_on_split(self, split):
        raise NotImplementedError
    
    def make_predictions(self):
        self.pred_flag = True
        self.prepare_for_test()
        
        os.makedirs(self.config.predictions_path, exist_ok=True)
        for split in ['trainset', 'valset', 'testset']:
            print('='*40)
            print(f'Prediction on {split}.')
            self.make_predictions_on_split(split)
            print()
    
    def do_evaluations(self):
        self.eval_flag = True
        self.prepare_for_test()
        
        os.makedirs(self.config.predictions_path, exist_ok=True)
        for split in ['trainset', 'valset', 'testset']:
            print('='*40)
            print(f'Evaluation on {split}.')
            self.do_evaluations_on_split(split)
            print()
        
