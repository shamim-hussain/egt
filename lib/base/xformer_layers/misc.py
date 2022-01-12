import tensorflow as tf
tfk = tf.keras

from .shaping import split_dim

class BiasAdd(tfk.layers.Layer):
    def __init__(self, initializer='zeros', regularizer=None, **kwargs):
        super().__init__(**kwargs)
        
        self.initializer = initializer
        self.regularizer = regularizer
        
        self.supports_masking = True
        
    def get_config(self):
        config = super().get_config().copy()
        config.update(dict(
            initializer = self.initializer,
            regularizer = self.regularizer
            ))
    
    def build(self, input_shape):
        self.b = self.add_weight(name='b', shape=[input_shape[-1]], 
                                    initializer=self.initializer,
                                    regularizer=self.regularizer)
        self.built = True
    
    def call(self, inputs):
        return inputs + self.b


class SplitDim(tfk.layers.Layer):
    def __init__(self, splits=2, axis=-1, **kwargs):
        super().__init__(**kwargs)

        self.splits = splits
        self.axis = axis
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            splits = self.splits,
            axis = self.axis,
        )
        return config
    
    def call(self, inputs):
        return split_dim(inputs, splits=self.splits, axis=self.axis)




class RandomNeg(tfk.layers.Layer):
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        if training:
            stat_shape = inputs.shape
            dyn_shape = tf.shape(inputs)

            uniform_s = tf.random.uniform(shape  = [dyn_shape[0], 1, dyn_shape[2], 1],
                                          minval = 0.,
                                          maxval = 1.,
                                          dtype  = tf.float32)
            signs = tf.where(uniform_s < 0.5, -1., 1.)
            signs.set_shape([stat_shape[0], 1, stat_shape[2], 1])

            outputs = inputs * signs
        else:
            outputs = inputs
        return outputs


class RandomNegEig(tfk.layers.Layer):
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        if training:
            stat_shape = inputs.shape
            dyn_shape = tf.shape(inputs)

            uniform_s = tf.random.uniform(shape  = [dyn_shape[0], 1, dyn_shape[2]],
                                          minval = 0.,
                                          maxval = 1.,
                                          dtype  = tf.float32)
            signs = tf.where(uniform_s < 0.5, -1., 1.)
            signs.set_shape([stat_shape[0], 1, stat_shape[2]])

            outputs = inputs * signs
        else:
            outputs = inputs
        return outputs


class RandomNegP(tfk.layers.Layer):
    def __init__(self, 
                 num_updates = 2000, 
                 min_prob    = 0., 
                 max_prob    = 0.5, 
                 **kwargs):
        self.num_updates = num_updates
        self.min_prob    = min_prob    
        self.max_prob    = max_prob    
        super().__init__(**kwargs)
    
    def get_config(self):
        return dict(
            num_updates = self.num_updates ,
            min_prob    = self.min_prob    ,
            max_prob    = self.max_prob    ,
            **super().get_config(),
        )
    
    def build(self, input_shape):
        self.inv_prob = self.add_weight(name='inv_prob', shape=[],
                                        initializer=tfk.initializers.Constant(value=self.min_prob), 
                                        trainable=False,
                                        aggregation=tf.VariableAggregation.MEAN)
        self.built = True
        
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        if training:
            stat_shape = inputs.shape
            dyn_shape = tf.shape(inputs)

            uniform_s = tf.random.uniform(shape  = [dyn_shape[0], 1, dyn_shape[2], 1],
                                          minval = 0.,
                                          maxval = 1.,
                                          dtype  = tf.float32)
            signs = tf.where(uniform_s < self.inv_prob, -1., 1.)
            signs.set_shape([stat_shape[0], 1, stat_shape[2], 1])

            outputs = inputs * signs
            
            new_prob = tf.minimum(self.max_prob, 
                                  self.inv_prob + (self.max_prob-self.min_prob)/self.num_updates)
            self.inv_prob.assign(new_prob)
        else:
            outputs = inputs
        return outputs




