
import tensorflow as tf
from tensorflow.keras import layers

from ...base.track_layers import CustomLayers
from ...base.xformer_layers.masking import Neg1MaskedEmbedding
from ..graph_xformer_model_base import GraphTransformerBase
from ..graph_model_base import AdjMatModel, SVDFeatModel, EigFeatModel


custom_layers = CustomLayers(Neg1MaskedEmbedding) 



class DCTransformer(AdjMatModel, GraphTransformerBase):
    def __init__(self                           ,
                 num_node_features  = 7         ,
                 num_target_labels  = 6         ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(     
            num_node_features  = num_node_features  ,
            num_target_labels  = num_target_labels  ,
        )
        self.tracked_layers.track_module(custom_layers)
    

    def get_node_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_node_inputs()
        inputs.update(
            nodef = layers.Input([config.max_length],
                                    name='node_features')
        )
        return inputs
    
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'nodef':
            x = layers.Neg1MaskedEmbedding(config.num_node_features+1, 
                                               config.model_width, 
                                               name='node_emb')(x)
        else:
            x = super().create_embedding(name, x)
            
        return x
    
    
    def readout_embeddings(self, h, e):
        config = self.config
        layers = self.tracked_layers

        h = self.mlp_out(h)
        h = layers.Dense(config.num_target_labels, name='target',
                          kernel_regularizer=self.l2reg)(h)
        return h



class DCSVDTransformer(SVDFeatModel, DCTransformer):
    pass

class DCEigTransformer(EigFeatModel, DCTransformer):
    pass