
import tensorflow as tf
from tensorflow.keras import metrics, backend

class MaskedXEntropy(metrics.Metric):
            def __init__(self, name='xent', **kwargs):
                super().__init__(name=name, **kwargs)
                self.total_xent = self.add_weight(name='xent', initializer='zeros')
                self.num_samples = self.add_weight(name='samples', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                assert y_true.shape.rank == 2
                y_true_shp = tf.shape(y_true)
                
                y_true = tf.reshape(y_true, shape=[-1])
                y_pred = tf.reshape(y_pred, shape=[-1])
                
                mask = tf.cast(y_true >= 0, tf.float32)
                y_true_p = tf.cast(tf.maximum(y_true, 0), tf.float32)
                
                xent = backend.binary_crossentropy(y_true_p, y_pred, 
                                                    from_logits=True)
                
                masked_xent = mask * xent
                
                new_xent = tf.reduce_sum(masked_xent)/tf.cast(y_true_shp[1], 
                                                              masked_xent.dtype)
                new_samples = tf.cast(y_true_shp[0], new_xent.dtype)
                
                self.total_xent.assign_add(new_xent)
                self.num_samples.assign_add(new_samples)

            def result(self):
                return tf.math.divide_no_nan(self.total_xent, self.num_samples)
            
            def reset_state(self):
                self.total_xent.assign(0)
                self.num_samples.assign(0)