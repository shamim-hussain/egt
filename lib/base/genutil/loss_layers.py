import tensorflow as tf
from tensorflow.keras import backend, layers, losses,initializers


class SparseXEntropy(layers.Layer):
    def __init__(self, multiplier,
                 from_logits=False,
                 mask_zero=False,
                 reduction='mean',
                 metric_name=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.multiplier  = multiplier
        self.from_logits = from_logits
        self.mask_zero   = mask_zero
        self.metric_name = metric_name
        self.reduction   = reduction
        
        self.supports_masking = True
    
    def get_config(self):
        return dict(
            **super().get_config(),
            multiplier  = self.multiplier,
            from_logits = self.from_logits,
            mask_zero   = self.mask_zero,
            metric_name = self.metric_name,
            reduction   = self.reduction,
        )
    
    def build(self, input_shape):
        self.multiplier_var = self.add_weight(name='multiplier', shape=[],
                                              initializer=initializers.Constant(self.multiplier),
                                              trainable=False, 
                                              aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.built = True
    
    def call(self, inputs, mask=None):
        y_true, y_pred, *others = inputs
        
        b_shape = tf.shape(y_true)[0]
        
        y_true_re = tf.reshape(y_true, [-1])
        y_pred_re = tf.reshape(y_pred, [-1,y_pred.shape[-1]])
        
        elem_loss = backend.sparse_categorical_crossentropy(y_true_re, y_pred_re, self.from_logits)
        
        if self.mask_zero:
            mask = tf.cast((y_true_re > 0), elem_loss.dtype)
            assert mask.shape.rank == elem_loss.shape.rank
            elem_loss = elem_loss * mask
        
        elem_loss = tf.reshape(elem_loss, [b_shape, -1])
        if self.reduction.lower() == 'mean':
            loss = tf.reduce_mean(elem_loss, axis=-1)
        elif self.reduction.lower() == 'sum':
            loss = tf.reduce_sum(elem_loss, axis=-1)
        else:
            raise ValueError
        
        if self.metric_name is not None:
            self.add_metric(loss, name=self.metric_name, 
                            aggregation='mean')
        
        self.add_loss(tf.reduce_mean(loss)*self.multiplier_var)
        
        return inputs
    
    def compute_mask(self, inputs, mask):
        return mask