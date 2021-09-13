import tensorflow as tf
from tensorflow.keras import layers, callbacks
import numpy as np

class GlobalStep(layers.Layer):
    def __init__(self, initial_step=0, **kwargs):
        super().__init__(**kwargs)
        self.initial_step = initial_step
        self.supports_masking = True
        
    def get_config(self):
        return dict(
            **super().get_config(),
            initial_step = self.initial_step
        )
        
    def build(self, input_shape):
        self.global_step = self.add_weight(name='global_step', shape=[], dtype=tf.int64,
                                           initializer=tf.keras.initializers.Constant(value=self.initial_step), 
                                           trainable=False,
                                           aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.built = True
        
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        if training:
            self.global_step.assign(self.global_step+1)
            outputs = inputs
        else:
            outputs = inputs
        
        return outputs
    
    def compute_mask(self, inputs, mask=None):
        return mask



class WarmUpAndCosine(callbacks.Callback):
    def __init__(self, warmup_steps, max_lr, min_lr = 0.,
                 total_steps = None,
                 layer_name = 'global_step_tracker',
                 **kwargs):
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.layer_name = layer_name
        
        self.lr_span = self.max_lr-self.min_lr
        self.wup_lr_incr = self.lr_span/self.warmup_steps
        if self.total_steps is not None:
            self.cosine_w = 0.5* np.pi / (self.total_steps-self.warmup_steps)
        
    def on_train_batch_begin(self, batch, logs=None):
        global_step, = self.model.get_layer(self.layer_name).get_weights()
        
        if global_step < self.warmup_steps:
            cur_lr = self.min_lr + self.wup_lr_incr*(global_step+1)
            self.model.optimizer.lr.assign(cur_lr)
        elif self.total_steps is not None:
            if global_step <= self.total_steps:
                cur_lr = self.min_lr + self.lr_span*np.cos(self.cosine_w*(global_step-self.warmup_steps))
                self.model.optimizer.lr.assign(cur_lr)
            else:
                self.model.stop_training = True
        
    def on_epoch_begin(self, epoch, logs=None):
        self.on_batch_begin(0)
        global_step, = self.model.get_layer(self.layer_name).get_weights()
        cur_lr = self.model.optimizer.lr.numpy()
        print(f'Current Global Step: {global_step}; Learning Rate: {cur_lr:.7f}', flush=True)
        