
import tensorflow as tf
from tensorflow.keras import layers

from ...base.track_layers import CustomLayers
from ...base.xformer_layers.pairwise_op import PairwiseOp
from ...base.xformer_layers.masking import MaskedGlobalAvgPooling2D
from ..graph_xformer_model_base import GraphTransformerBase
from ..graph_model_base import AdjMatModel, SVDFeatModel, VNModel


custom_layers = CustomLayers(MaskedGlobalAvgPooling2D, PairwiseOp)



class DCTransformer(VNModel, AdjMatModel, GraphTransformerBase):
    def __init__(self                                               ,
                 node_feat_dims     = (53, 3, 7, 10, 5, 5, 6, 2, 2) ,
                 edge_feat_dims     = (4, 3, 2)                     ,
                 mask_value         = -1                            ,
                 readout_edges      = False                         ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(        
            node_feat_dims     = node_feat_dims     ,
            edge_feat_dims     = edge_feat_dims     ,
            mask_value         = mask_value         ,
            readout_edges      = readout_edges      ,
        )
        self.tracked_layers.track_module(custom_layers)
    

    def get_node_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_node_inputs()
        inputs.update(
            nodef = layers.Input([config.max_length, len(config.node_feat_dims)],
                                 dtype='int32', name='node_features')
        )
        
        return inputs


    def get_edge_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_edge_inputs()
        inputs.update(
            fmat = layers.Input([config.max_length, config.max_length, 
                                 len(config.edge_feat_dims)],
                                dtype='int32', name='feature_matrix'),
        )
        return inputs
    
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'nodef':
            node_ds = config.node_feat_dims
            def node_stack(node_f):
                node_fs = tf.unstack(node_f, axis=-1)
                oh_vecs = []
                for feat, dim in zip(node_fs, node_ds):
                    oh_vecs.append(tf.one_hot(feat,dim,dtype=tf.float32))
                node_oh = tf.concat(oh_vecs, axis=-1)
                return node_oh
            def compute_mask(inputs, mask):
                return mask
            x = layers.Masking(mask_value=config.mask_value, name='node_mask')(x)
            x = layers.Lambda(node_stack, mask=compute_mask, name='node_feat_oh')(x)
            x = layers.Dense(config.model_width, name='node_emb',
                             kernel_initializer='uniform',
                             kernel_regularizer=self.l2reg)(x)

        elif name == 'fmat':
            edge_ds = config.edge_feat_dims
            def edge_stack(edge_f):
                edge_fs = tf.unstack(edge_f, axis=-1)
                oh_vecs = []
                for feat, dim in zip(edge_fs, edge_ds):
                    oh_vecs.append(tf.one_hot(feat,dim,dtype=tf.float32))
                edge_oh = tf.concat(oh_vecs, axis=-1)
                return edge_oh
            def compute_mask(inputs, mask):
                return mask
            x = layers.Masking(mask_value=config.mask_value, name='edge_mask')(x)
            x = layers.Lambda(edge_stack, mask=compute_mask, name='edge_feat_oh')(x)
            x = layers.Dense(config.edge_width, name='edge_emb',
                             kernel_initializer='uniform',
                             kernel_regularizer=self.l2reg)(x)
            
        else:
            x = super().create_embedding(name, x)
        
        return x


    def readout_embeddings(self, h, e):
        config = self.config
        layers = self.tracked_layers

        if config.num_virtual_nodes > 0:
            h = layers.GetVirtualNodes(num_nodes=config.num_virtual_nodes,
                                       name='get_virtual_nodes')(h)
            h = layers.Flatten(name='virtual_nodes_flatten')(h)
        else:
            h = layers.GlobalAveragePooling1D(name='node_glob_avg_pool')(h)
        
        x = h
        if config.readout_edges:
            e = layers.MaskedGlobalAvgPooling2D(name='edge_glob_avg_pool')(e)
            x = layers.Concatenate(name='cat_node_and_edge_out')([x,e])
        
        x = self.mlp_out(x)
        
        def grad_relu(x):
            return x + tf.stop_gradient(tf.nn.relu(x) - x)
        
        x = layers.Dense(1, name='target', activation=grad_relu,
                          kernel_regularizer=self.l2reg)(x)
        return x



class DCSVDTransformer(SVDFeatModel, DCTransformer):
    pass


