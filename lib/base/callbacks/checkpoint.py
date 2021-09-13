
import tensorflow as tf
tfk=tf.keras
import numpy as np
import os


class CheckpointCallback(tfk.callbacks.Callback):
    def __init__(self, save_path, model, 
                 state         = None  , 
                 on_batch_end  = None  ,
                 on_epoch_end  = None  ,
                 relod_on_nan  = False , 
                 verbose       = 1     ,
                 ):
        self.model           = model
        self.state           = state or {}
        self.save_path       = save_path
        self.on_batch_end_fn = on_batch_end
        self.on_epoch_end_fn = on_epoch_end
        self.relod_on_nan    = relod_on_nan
        self.verbose         = verbose

        if self.verbose:
            self.print_fn = lambda x: print(x, flush=True)
        else:
            self.print_fn = lambda x: x

        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.model.optimizer,
                                              **self.state)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=self.save_path,
                                                             max_to_keep=1)
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        
        if self.relod_on_nan:
            loss = logs.get('loss', 0.)
            if np.isnan(loss) or np.isinf(loss):
                self.print_fn(f'Batch {batch}: Invalid loss, reloading checkpoint!!!')
                self.load_checkpoint()
                return
                
        if self.on_batch_end_fn is not None:
            if isinstance(self.on_batch_end_fn,list):
                for fn in self.on_batch_end_fn:
                    fn(self.model, self.state, batch, logs)
            else:
                self.on_batch_end_fn(self.model, self.state, batch, logs)
            

    def load_checkpoint(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        ret = self.checkpoint_manager.latest_checkpoint
        if ret: self.print_fn(f'Checkpoint loaded from {self.save_path}')
        return ret
    
    def save_checkpoint(self):
        self.checkpoint_manager.save()
        self.print_fn(f'Checkpoint saved to {self.save_path}')
    
    def on_train_begin(self,epoch,logs=None):
        self.load_checkpoint()
    
    def on_epoch_end(self,epoch,logs=None):
        logs = logs or {}
        
        if self.relod_on_nan:
            loss = logs.get('loss', 0.)
            if np.isnan(loss) or np.isinf(loss):
                self.print_fn('Invalid loss, checkpoint not saved!')

        if self.on_epoch_end_fn is not None:
            if isinstance(self.on_epoch_end_fn,list):
                for fn in self.on_epoch_end_fn:
                    fn(self.model, self.state, epoch, logs)
            else:
                self.on_epoch_end_fn(self.model, self.state, epoch, logs)
        
        self.print_fn(f'\nCHECKPOINT Epoch: {epoch+1}')
        self.save_checkpoint()


class SaveWhenCallback(tfk.callbacks.Callback):
    def __init__(self, save_path,
                 when              = 'epoch;True;epoch{epoch:0>4d}',
                 state             = None  , 
                 verbose           = 1     ,
                 save_weights_only = True  ,
                 ):
        self.save_path         = save_path
        self.when              = when
        self.state             = state or {}
        self.verbose           = verbose          
        self.save_weights_only = save_weights_only
        
        if self.verbose:
            self.print_fn = lambda x: print(x, flush=True)
        else:
            self.print_fn = lambda x: x
        
        self.criterions = []
        for item in self.when.split('#'):
            elems = item.split(';')
            event = elems[0].strip().lower()
            cond  = elems[1].strip()
            format= elems[2].strip()
            self.criterions.append( (event, cond, format) )
    
    def save_on_event(self, event, scope):
        for e, c, f in self.criterions:
            if e == event:
                try:
                    if eval(c,scope):
                        save_file = os.path.join(self.save_path, f.format(**scope)+'.h5')
                        if not self.save_weights_only:
                            self.model.save(save_file)
                        else:
                            self.model.save_weights(save_file)
                        self.print_fn(f'\nSAVE:{e};{c}: model saved to {save_file}')
                except NameError:
                    self.print_fn(f'\nSAVE:{e};{c}: did not find log, IGNORING')
    
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        scope = logs.copy()
        scope['batch'] = batch
        scope.update(dict( (k,v.numpy().tolist()) for k,v in self.state.items()))
        self.save_on_event('batch', scope)
    
    def on_epoch_end(self, epoch, logs):
        logs = logs or {}
        scope = logs.copy()
        scope['epoch'] = epoch+1
        scope.update(dict( (k,v.numpy().tolist()) for k,v in self.state.items()))
        self.save_on_event('epoch', scope)



