import tensorflow as tf
tfk = tf.keras


class Neg1MaskedEmbedding(tfk.layers.Embedding):
    def __init__(self                               ,
                 input_dim                          ,
                 output_dim                         ,
                 embeddings_initializer = 'uniform' ,
                 embeddings_regularizer = None      ,
                 activity_regularizer   = None      ,
                 embeddings_constraint  = None      ,
                 input_length           = None      ,
                 return_mask            = False     ,   
                 **kwargs):
        super().__init__(input_dim              = input_dim             ,
                         output_dim             = output_dim            ,
                         embeddings_initializer = embeddings_initializer,
                         embeddings_regularizer = embeddings_regularizer,
                         activity_regularizer   = activity_regularizer  ,
                         embeddings_constraint  = embeddings_constraint ,
                         mask_zero              = True                  ,
                         input_length           = input_length          ,
                         **kwargs                                        )
        self.return_mask = return_mask
    
    def get_config(self):
        config = super().get_config().copy()
        config.pop('mask_zero')
        config.update(
            return_mask = self.return_mask,
        )
        return config
    
    def call(self, inputs):
        if not self.return_mask:
            return super().call(inputs+1)
        else:
            mask = tf.cast(self.compute_mask(inputs), tf.float32)
            return super().call(inputs+1), mask

    def compute_mask(self, inputs, mask=None):
        return super().compute_mask(inputs+1, mask)


class MaskedSparseXELoss(tfk.layers.Layer):
    def __init__(self, 
                from_logits = True  ,
                reduction   = 'mean',
                return_loss = False ,
                **kwargs):
        super().__init__(**kwargs)
        self.from_logits    = from_logits
        self.reduction      = reduction
        self.return_loss    = return_loss

        self.supports_masking = True
    
    def get_config(self):
        config = dict(
            **super().get_config()          ,
            from_logits = self.from_logits  ,
            return_loss = self.return_loss  ,
            reduction   = self.reduction    ,
            )
        return config

    def call(self, inputs, mask=None):
        if len(inputs) == 3:
            y_true, y_pred, mask_in = inputs
            if mask is not None:
                mask = tf.cast(mask, y_pred.dtype) * mask_in
            else:
                mask = mask_in
        else:
            y_true, y_pred = inputs
            mask = tf.cast(mask, y_pred.dtype)
        
        losses_elem = mask * \
                      tfk.backend.sparse_categorical_crossentropy(y_true, y_pred,
                                                      from_logits=self.from_logits)

        reduction_dims = tuple(range(1,losses_elem.shape.rank))
        if self.reduction == 'sum':
            loss = tf.reduce_sum(losses_elem, axis=reduction_dims)
            self.add_loss(tf.reduce_mean(loss))
        elif self.reduction == 'masked_mean':
            loss = tf.reduce_sum(losses_elem, axis=reduction_dims)
            loss = loss/tf.reduce_sum(mask, axis=reduction_dims)
            self.add_loss(tf.reduce_mean(loss))
        elif self.reduction == 'mean':
            self.add_loss(tf.reduce_mean(losses_elem))
        else:
            raise ValueError('Reduction must be sum/mean')

        
        if self.return_loss:
            return y_pred, loss
        else:
            return y_pred

class MaskedGlobalAvgPooling2D(tfk.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
    
    def get_config(self):
        return super().get_config()
    
    def call(self, inputs, mask):
        if mask is None:
            raise ValueError
        
        assert mask.shape.rank == 3
        assert inputs.shape.rank == 4
        
        mask = tf.cast(mask, dtype=inputs.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        
        sum_inputs = tf.reduce_sum(inputs*mask, axis=(1,2))
        sum_mask = tf.reduce_sum(mask, axis=(1,2))
        
        avg_input = tf.math.divide_no_nan(sum_inputs, sum_mask)
        return avg_input
    
    def compute_mask(self, inputs, mask):
        return None