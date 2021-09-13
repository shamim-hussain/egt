
import tensorflow as tf
from tensorflow.keras import backend, losses, metrics
import numpy as np

def weighted_sparse_xentropy(y_true, y_pred, 
                             weights,
                             from_logits=False):
    tshp = tf.shape(y_true)
    tshp_stat = y_true.shape
    
    y_true = tf.reshape(y_true, shape=[-1])
    y_pred = tf.reshape(y_pred, shape=[-1, y_pred.shape[-1]])
    
    weights = tf.gather(weights, tf.cast(y_true, tf.int32))
    xent = backend.sparse_categorical_crossentropy(y_true, y_pred, 
                                                    from_logits=from_logits, 
                                                    axis=-1)
    assert weights.shape.rank == xent.shape.rank
    w_xent = weights * xent
    
    w_xent = tf.reshape(w_xent, tshp)
    w_xent.set_shape(tshp_stat)
    return w_xent


class WeightedSparseXEntropyLoss(losses.Loss):
    def __init__(self,  
                 class_weights = None, 
                 class_sizes   = None,
                 from_logits   = False, 
                 name          = 'xent', 
                 **kwargs):
        super().__init__(
            name = name , 
            **kwargs
        )
        self.class_weights = class_weights
        self.class_sizes   = class_sizes  
        self.from_logits   = from_logits  
        
        if class_weights is None:
            if class_sizes is None:
                raise ValueError
            class_sizes   = np.array(class_sizes, dtype='float32')
            class_weights = class_sizes.sum() - class_sizes
            class_weights = class_weights/class_weights.sum()
            self._class_weights = tf.constant(class_weights, tf.float32)
        else:
            self._class_weights = tf.constant(class_weights, tf.float32)
    
    def get_config(self):
        return dict(
            **super().get_config(),
            class_weights = self.class_weights,
            class_sizes   = self.class_sizes  ,
            from_logits   = self.from_logits  ,
        )
        
    def call(self, y_true, y_pred):
        return weighted_sparse_xentropy(y_true, y_pred,
                                        weights = self._class_weights,
                                        from_logits   = self.from_logits   )

class WeightedSparseXEntropyMetric(metrics.Metric):
    def __init__(self,  
                 class_weights = None, 
                 class_sizes   = None,
                 from_logits   = False, 
                 name          = 'xent', 
                 **kwargs):
        super().__init__(
            name = name , 
            **kwargs
        )
        self.class_weights = class_weights
        self.class_sizes   = class_sizes  
        self.from_logits   = from_logits  
        
        if class_weights is None:
            if class_sizes is None:
                raise ValueError
            class_sizes   = np.array(class_sizes, dtype='float32')
            class_weights = class_sizes.sum() - class_sizes
            class_weights = class_weights/class_weights.sum()
            self._class_weights = tf.constant(class_weights, tf.float32)
        else:
            self._class_weights = tf.constant(class_weights, tf.float32)
        
        self.total = self.add_weight(name='total', initializer='zeros') 
        self.count = self.add_weight(name='count', initializer='zeros')  
    
    def get_config(self):
        return dict(
            **super().get_config(),
            class_weights = self.class_weights,
            class_sizes   = self.class_sizes  ,
            from_logits   = self.from_logits  ,
        )
    
    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)
    
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = weighted_sparse_xentropy(y_true, y_pred,
                                        weights = self._class_weights,
                                        from_logits = self.from_logits   )
        
        if sample_weight is None:
            self.total.assign_add(tf.reduce_sum(loss))
            self.count.assing_add(tf.reduce_prod(tf.cast(tf.shape(loss),self.count.dtype)))
        else:
            if sample_weight.shape.rank > loss.shape.rank:
                sample_weight = tf.squeeze(sample_weight)
            if loss.shape.rank > sample_weight.shape.rank:
                loss = tf.squeeze(loss)
            
            sample_weight = tf.cast(sample_weight,self.count.dtype)
            self.total.assign_add(tf.reduce_sum(loss*sample_weight))
            self.count.assign_add(tf.reduce_sum(sample_weight))
            