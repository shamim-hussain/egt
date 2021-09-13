
import tensorflow as tf
from tensorflow.keras import layers

from ...base.track_layers import CustomLayers
from ...base.xformer_layers.pairwise_op import PairwiseOp
from ...base.xformer_layers.masking import MaskedGlobalAvgPooling2D
from ..graph_xformer_model_base import GraphTransformerBase
from ..graph_model_base import AdjMatModel, SVDFeatModel, VNModel


custom_layers = CustomLayers(MaskedGlobalAvgPooling2D, PairwiseOp)



class DCTransformer(VNModel, AdjMatModel, GraphTransformerBase):
    def __init__(self                           ,
                 num_node_features  = 5         ,
                 num_edge_features  = 1         ,
                 num_target_labels  = 10        ,
                 mask_value         = -1.       ,
                 readout_edges      = False     ,
                 node2edge_embed    = False     ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(        
            num_node_features  = num_node_features  ,
            num_edge_features  = num_edge_features  ,
            num_target_labels  = num_target_labels  ,
            mask_value         = mask_value         ,
            readout_edges      = readout_edges      ,
            node2edge_embed    = node2edge_embed    ,
        )
        self.tracked_layers.track_module(custom_layers)
    

    def get_node_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_node_inputs()
        inputs.update(
            nodef = layers.Input([config.max_length, config.num_node_features],
                                    name='node_features')
        )
        
        return inputs


    def get_edge_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_edge_inputs()
        inputs.update(
            fmat = layers.Input([config.max_length, config.max_length, 
                                 config.num_edge_features],
                                        name='feature_matrix'),
        )
        return inputs
    
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'nodef':
            x = layers.Masking(mask_value=config.mask_value, name='node_mask')(x)
            x = layers.Dense(config.model_width, name='node_emb',
                             kernel_regularizer=self.l2reg)(x)

        elif name == 'fmat':
            x = layers.Masking(mask_value=config.mask_value, name='edge_mask')(x)
            x = layers.Dense(config.edge_width, name='edge_emb',
                             kernel_regularizer=self.l2reg)(x)
            
        else:
            x = super().create_embedding(name, x)
        
        return x
    
    
    def update_across_embeddings(self, node_inputs, edge_inputs,
                                 node_embeddings, edge_embeddings):
        layers = self.tracked_layers
        config = self.config
        
        if config.node2edge_embed:
            x = node_inputs['nodef']
            x = layers.Dense(config.edge_width*2, name='node_edge_emb',
                             kernel_regularizer=self.l2reg)(x)
            x = layers.PairwiseOp(op='addsub',
                                  op_kwargs=dict(add=True,sub=False),
                                  split_axis=-1,
                                  name='node2edge_pairwise')(x)
            edge_embeddings.update(node2edge=x)

        return super().update_across_embeddings(node_inputs, edge_inputs,
                                                node_embeddings, edge_embeddings)


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
        x = layers.Dense(config.num_target_labels, name='target',
                         kernel_regularizer=self.l2reg)(x)
        return x



class DCSVDTransformer(SVDFeatModel, DCTransformer):
    pass


