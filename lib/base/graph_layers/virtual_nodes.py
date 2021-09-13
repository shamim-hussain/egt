import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.training.tracking.base import no_automatic_dependency_tracking_scope


class VirtualNodeEmbedding(layers.Layer):
    def __init__(self, 
                 num_nodes    = 1         ,
                 initializer  = 'uniform' ,
                 regularizer  = None      ,
                 constraint  = None      ,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_nodes    = num_nodes  
        self.initializer  = initializer
        self.regularizer  = regularizer
        self.constraint   = constraint 
    
    def get_config(self):
        config = super().get_config()
        config.update(
            num_nodes    = self.num_nodes  ,
            initializer  = self.initializer,
            regularizer  = self.regularizer,
            constraint   = self.constraint ,
        )
        return config
    
    def build(self, input_shape):
        self.embeddings = self.add_weight(name        = 'virtual_node_embeddings',
                                          shape       = [self.num_nodes, input_shape[-1]],
                                          initializer = self.initializer,
                                          regularizer = self.regularizer,
                                          constraint  = self.constraint,
                                          )
                        
        self.built = True
    
    def call(self, inputs, mask=None):
        tiled_embeddings = tf.tile(tf.expand_dims(self.embeddings, axis=0), 
                                   [tf.shape(inputs)[0], 1, 1])
        outputs = tf.concat([tiled_embeddings, inputs], axis=1)
        return outputs
    
    def compute_mask(self, inputs, mask):
        new_true = tf.ones([tf.shape(mask)[0], self.num_nodes], dtype=tf.bool)
        new_mask = tf.concat([new_true, mask], axis=1)
        return new_mask


class VirtualEdgeEmbedding(layers.Layer):
    def __init__(self, 
                 num_nodes    = 1         ,
                 initializer  = 'uniform' ,
                 regularizer  = None      ,
                 constraint   = None      ,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_nodes    = num_nodes  
        self.initializer  = initializer
        self.regularizer  = regularizer
        self.constraint   = constraint 
    
    def get_config(self):
        config = super().get_config()
        config.update(
            num_nodes    = self.num_nodes  ,
            initializer  = self.initializer,
            regularizer  = self.regularizer,
            constraint   = self.constraint ,
        )
        return config
    
    def build(self, input_shape):
        self.embeddings = self.add_weight(name        = 'virtual_edge_embeddings',
                                          shape       = [self.num_nodes, input_shape[-1]],
                                          initializer = self.initializer,
                                          regularizer = self.regularizer,
                                          constraint  = self.constraint,
                                          )
                        
        self.built = True
    
    def call(self, inputs, mask=None):
        bshape_d, eshape1_d, eshape2_d, _ = tf.unstack(tf.shape(inputs))
        
        emb_r, emb_c = self.embeddings[None,:,None,:], self.embeddings[None,None,:,:]
        tiled_row_embeddings = tf.tile(emb_r, [bshape_d, 1, eshape2_d, 1])
        tiled_col_embeddings = tf.tile(emb_c, [bshape_d, eshape1_d, 1, 1])
        box_embeddings = 0.5*(emb_r+emb_c)
        tiled_box_embeddings = tf.tile(box_embeddings, [bshape_d, 1, 1, 1])
        
        outputs = inputs
        outputs = tf.concat([tiled_row_embeddings, outputs], axis=1)
        bc_emb = tf.concat([tiled_box_embeddings, tiled_col_embeddings], axis=1)
        outputs = tf.concat([bc_emb, outputs], axis=2)
        return outputs
    
    def compute_mask(self, inputs, mask):
        bshape_d, eshape1_d, eshape2_d = tf.unstack(tf.shape(mask))
        
        row_true = tf.ones([bshape_d, self.num_nodes, eshape2_d], dtype=tf.bool)
        col_true = tf.ones([bshape_d, eshape1_d+self.num_nodes, self.num_nodes], dtype=tf.bool)
        
        new_mask = mask
        new_mask = tf.concat([row_true, new_mask], axis=1)
        new_mask = tf.concat([col_true, new_mask], axis=2)
        return new_mask


class GetVirtualNodes(layers.Layer):
    def __init__(self, 
                 num_nodes    = 1 ,
                 mask_out     = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_nodes    = num_nodes 
        self.mask_out     = mask_out 
    
    def get_config(self):
        config = super().get_config()
        config.update(
            num_nodes    = self.num_nodes,
            mask_out     = self.mask_out
        )
        return config
    
    def call(self, inputs, mask=None):
        outputs = inputs[:,:self.num_nodes,:]
        outputs.set_shape([None, self.num_nodes, inputs.shape[2]])
        return outputs
    
    def compute_mask(self, inputs, mask):
        if self.mask_out:
            new_mask = mask[:,:self.num_nodes]
            new_mask.set_shape([None, self.num_nodes])
            return new_mask
        else:
            return None