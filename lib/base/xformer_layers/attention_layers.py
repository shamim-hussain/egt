
import tensorflow as tf
tfk = tf.keras

from .attention import *
from .shaping import *

class Attention(tfk.layers.Layer):
    def __init__(self,
                 num_heads           = 8,
                 causal              = False,
                 splits              = 3,
                 pad                 = False,
                 scale_logits        = True,
                 clip_logits_value   = None,
                 attn_mask           = False,
                 logits_scaler       = False,
                 logits_bias         = False,
                 attention_scaler    = False,
                 merge_heads         = None,
                 return_logits       = False,
                 return_matrix       = False,
                 headed_input        = False,
                 headed_output       = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking=True
        
        self.num_heads           = num_heads           
        self.causal              = causal              
        self.splits              = splits              
        self.pad                 = pad                 
        self.scale_logits        = scale_logits        
        self.clip_logits_value   = clip_logits_value   
        self.attn_mask           = attn_mask      
        self.logits_scaler       = logits_scaler       
        self.logits_bias         = logits_bias         
        self.attention_scaler    = attention_scaler    
        self.merge_heads         = merge_heads     
        self.return_logits       = return_logits       
        self.return_matrix       = return_matrix       
        self.headed_input        = headed_input        
        self.headed_output       = headed_output       


    def get_config(self):
        config = super().get_config()
        config.update(
            num_heads           = self.num_heads           ,
            causal              = self.causal              ,
            splits              = self.splits              ,
            pad                 = self.pad                 ,
            scale_logits        = self.scale_logits        ,
            clip_logits_value   = self.clip_logits_value   ,
            attn_mask           = self.attn_mask           ,
            logits_scaler       = self.logits_scaler       ,
            logits_bias         = self.logits_bias         ,
            attention_scaler    = self.attention_scaler    ,
            merge_heads         = self.merge_heads         ,
            return_logits       = self.return_logits       ,
            return_matrix       = self.return_matrix       ,
            headed_input        = self.headed_input        ,
            headed_output       = self.headed_output       ,
        )
        return config
    
    
    def call(self, inputs, mask=None):
        attn_scale_factor = None
        if self.attention_scaler:
            *inputs, attn_scale_factor = inputs

        scale_factor = None
        if self.logits_scaler:
            *inputs, scale_factor = inputs

        bias = None
        if self.logits_bias:
            *inputs, bias = inputs
        
        attn_mask = None
        if self.attn_mask:
            *inputs, attn_mask = inputs
        
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs, = inputs
        
        split_inputs = []
        if not isinstance(inputs, list):
            if not self.headed_input:
                inputs = create_heads(inputs, self.num_heads)
            split_inputs = split_dim(inputs, self.splits)
        else:
            for inp, split in zip(inputs, self.splits):
                if not self.headed_input:
                    inp = create_heads(inp, self.num_heads)
                split_inputs.extend(split_dim(inp, split))
        
        query, key, value = split_inputs
        
        if isinstance(mask, list):
            mask = mask[0]

        if self.merge_heads is not None:
            value = flatten_heads(value)
            
        outputs = dot_product_attention(query             = query, 
                                        key               = key, 
                                        value             = value, 
                                        mask              = mask, 
                                        attn_mask         = attn_mask,
                                        scale_factor      = scale_factor, 
                                        bias              = bias,
                                        attn_scale_factor = attn_scale_factor,
                                        scale_logits      = self.scale_logits,
                                        clip_logits_value = self.clip_logits_value,
                                        causal            = self.causal,
                                        pad               = self.pad,
                                        merge_heads       = self.merge_heads,
                                        return_logits     = self.return_logits,
                                        return_matrix     = self.return_matrix
                                        )
        
        # Flatten heads if necessary
        if not (self.headed_output or 
                (self.merge_heads is not None)):
            if not isinstance(outputs, tuple):
                outputs = flatten_heads(outputs)
            else:
                outp = flatten_heads(outputs[0])
                outputs = (outp,) + outputs[1:]
        return outputs


    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = [mask[0]]+[None]*(len(inputs)-1)
        return mask