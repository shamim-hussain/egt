
import tensorflow as tf
from tensorflow.keras import layers

from ...base.track_layers import CustomLayers
from ...base.xformer_layers.pairwise_op import PairwiseOp
from ..graph_xformer_model_base import GraphTransformerBase
from ..graph_model_base import AdjMatModel, SVDFeatModel


custom_layers = CustomLayers(PairwiseOp)



class DCTransformer(AdjMatModel, GraphTransformerBase):
    def __init__(self                           ,
                 num_node_features   = 2         ,
                 num_edge_features   = 1         ,
                 num_target_labels   = 2         ,
                 mask_value          = -1.       ,
                 use_node_embeddings = False     ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(        
            num_node_features   = num_node_features   ,        
            num_edge_features   = num_edge_features   ,
            num_target_labels   = num_target_labels   ,
            mask_value          = mask_value          ,
            use_node_embeddings = use_node_embeddings ,
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
    
    def readout_embeddings(self, h, e):
        config = self.config
        layers = self.tracked_layers

        if not config.use_node_embeddings:
            e = self.mlp_out(e)
            e = layers.Dense(config.num_target_labels, name='target',
                            kernel_regularizer=self.l2reg)(e)
            return e

        else:
            he = layers.PairwiseOp(op='cat', name='pairwise_node_emb')([h,h])
            he = layers.Concatenate(axis=-1, name='he_cat')([he,e])
            he = self.mlp_out(he)
            he = layers.Dense(config.num_target_labels, name='target',
                            kernel_regularizer=self.l2reg)(he)
            return he



class DCSVDTransformer(SVDFeatModel, DCTransformer):
    pass


