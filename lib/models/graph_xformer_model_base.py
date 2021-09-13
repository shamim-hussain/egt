
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from types import SimpleNamespace

from tensorflow.python.keras.backend import squeeze

from ..base.xformer_layers.attention_layers import Attention
from ..base.track_layers import TrackedLayers, CustomLayers
from ..base.genutil.warmup import GlobalStep

from .analysis import Analysis

custom_layers = CustomLayers(Attention, GlobalStep)



class GraphTransformerBase:
    def __init__(self                           ,
                 model_width        = 128       ,
                 edge_width         = 32        ,
                 num_heads          = 8         ,
                 max_length         = None      ,
                 pad_attention      = False     ,
                 merge_heads        = None      ,
                 gate_attention     = True      ,
                 dc_gates           = False     ,
                 model_height       = 4         ,
                 node_normalization = 'layer'   ,
                 edge_normalization = 'layer'   ,
                 l2_reg             = 0         ,
                 node_dropout       = 0         ,
                 edge_dropout       = 0         ,
                 add_n_norm         = False     ,
                 activation         = 'elu'     ,
                 mlp_layers         = [.5, .25] ,
                 do_final_norm      = True      ,
                 clip_logits_value  = [-5,5]    ,
                 edge_activation    = None      ,
                 edge_channel_type  = 'residual',
                 combine_layer_repr = False     ,
                 ffn_multiplier     = 2.        ,
                 node2edge_xtalk    = 0.        ,
                 edge2node_xtalk    = 0.        ,
                 global_step_layer  = False     ,
                 ):
        self.config = SimpleNamespace(
            model_width        = model_width        ,
            edge_width         = edge_width         ,
            num_heads          = num_heads          ,
            max_length         = max_length         ,
            pad_attention      = pad_attention      ,
            merge_heads        = merge_heads        ,
            gate_attention     = gate_attention     ,
            dc_gates           = dc_gates           ,
            model_height       = model_height       ,
            node_normalization = node_normalization ,
            edge_normalization = edge_normalization ,
            l2_reg             = l2_reg             ,
            node_dropout       = node_dropout       ,
            edge_dropout       = edge_dropout       ,
            add_n_norm         = add_n_norm         ,
            activation         = activation         , 
            mlp_layers         = mlp_layers         ,
            do_final_norm      = do_final_norm      ,
            edge_channel_type  = edge_channel_type  ,
            clip_logits_value  = clip_logits_value  ,
            edge_activation    = edge_activation    ,
            combine_layer_repr = combine_layer_repr ,
            ffn_multiplier     = ffn_multiplier     ,
            node2edge_xtalk    = node2edge_xtalk    ,
            edge2node_xtalk    = edge2node_xtalk    ,
            global_step_layer  = global_step_layer  ,
        )
        self.tracked_layers = TrackedLayers(custom_layers, layers)

        self.l2reg  = regularizers.l2(self.config.l2_reg) \
                                       if self.config.l2_reg > 0 else None
        # Analysis
        self.analysis = Analysis()


    def transform_embeddings(self, emb_nodes, emb_edges, edge_mask, return_all=False):
        config = self.config
        layers = self.tracked_layers
        
        all_node_repr = {}
        all_edge_repr = {}

        l2reg = self.l2reg
        
        norm_dict = dict(
            layer = layers.LayerNormalization,
            batch = layers.BatchNormalization
        )
        
        normlr_node = norm_dict[config.node_normalization]
        normlr_edge = norm_dict[config.edge_normalization]

        # MHA 
        def mha_block(tag, h, e, gates=None):
            y = h
            if not config.add_n_norm:
                h = normlr_node(name=f'norm_mha_{tag}')(h)            
            
            all_node_repr[tag] = h
            
            h = layers.Dense(config.model_width*3, name=f'dense_qkv_{tag}',
                             kernel_regularizer=l2reg)(h)
            
            if (gates is not None) and config.dc_gates:
                g_e = gates
                g_h = layers.Dense(config.num_heads*2, name=f'dense_node_gates_{tag}',
                                   kernel_regularizer=l2reg)(h)
                nh_t = config.num_heads
                g = layers.Lambda(lambda ee_hh: ee_hh[0] + ee_hh[1][:,:,None,:nh_t] + ee_hh[1][:,None,:,nh_t:],
                                  name=f'add_node_edge_gates_{tag}')([g_e,g_h])
                gates = layers.Activation('sigmoid', name=f'gate_sigmoid_{tag}')(g)
            
            with layers.namespace(f'mha'):
                h, e, mat = layers.Attention(num_heads        = config.num_heads,
                                            pad               = config.pad_attention,
                                            merge_heads       = config.merge_heads,
                                            attn_mask         = (edge_mask is not None),
                                            logits_bias       = (self.config.edge_channel_type != 'none'),
                                            return_logits     = True,
                                            clip_logits_value = config.clip_logits_value,
                                            attention_scaler  = (gates is not None),
                                            return_matrix     = True,
                                            name              = f'mha_{tag}'
                                            )([h] +
                                            ( [edge_mask] if edge_mask is not None                   else [] )+
                                            ( [e]         if self.config.edge_channel_type != 'none' else [] )+
                                            ( [gates]     if gates is not None                       else [] ))
    
                # Analysis
                self.analysis.add_analysis(f'mha_{tag}', e=e, mat=mat)
            
            h = layers.Dense(config.model_width, name=f'dense_mha_{tag}',
                             kernel_regularizer=l2reg)(h)
            if config.node_dropout > 0:
                h=layers.Dropout(config.node_dropout, name=f'drp_mha_{tag}')(h)
            h = layers.Add(name=f'res_mha_{tag}')([h,y])

            if config.add_n_norm:
                h = normlr_node(name=f'norm_mha_{tag}')(h) 
            
            return h, e
        
        
        # Edge Updates
        def edge_channel_contrib(tag, e):
            if config.edge_activation is not None and \
                config.edge_activation.lower().startswith('lrelu'):
                alpha = float(config.edge_activation[-1])/10
                e = layers.Dense(config.num_heads, name=f'dense_edge_b_{tag}',
                                 activation = None,
                                 kernel_regularizer=l2reg)(e)
                e = layers.LeakyReLU(alpha=alpha, name=f'lrelu_edge_b_{tag}')(e)
                
            else:
                e = layers.Dense(config.num_heads, name=f'dense_edge_b_{tag}',
                                 activation = config.edge_activation,
                                 kernel_regularizer=l2reg)(e)
            return e
        
        def edge_update_none(tag, h, e):
            gates = None
            # Analysis
            self.analysis.add_analysis(f'dense_edge_b_{tag}', e=e)
            
            h, _ = mha_block(tag, h, e, gates)
            
            return h, e

        def edge_update_bias(tag, h, e):
            e0 = e
            gates = None
            if config.gate_attention:
                if config.merge_heads is None:
                    gates = layers.Dense(config.num_heads,
                                         activation='sigmoid',
                                         name=f'attention_gates_{tag}',
                                         kernel_regularizer=l2reg)(e)
                else:
                    gates = layers.Dense(1,
                                         activation='sigmoid',
                                         name=f'attention_gates_{tag}',
                                         kernel_regularizer=l2reg)(e)
                    gates = layers.Lambda(lambda tensor: tf.squeeze(tensor, axis=-1),
                                          name=f'squeeze_gates_{tag}')(gates)
                # Analysis
                self.analysis.add_analysis(f'attention_gates_{tag}', gates=gates)
            
            e = edge_channel_contrib(tag, e)
            # Analysis
            self.analysis.add_analysis(f'dense_edge_b_{tag}', e=e)
            
            h, e = mha_block(tag, h, e, gates)
            
            return h, e0
        
        def edge_update_residual(tag, h, e):
            y = e
            if not config.add_n_norm:
                e = normlr_edge(name=f'norm_edge_{tag}')(e)
            
            all_edge_repr[tag] = e

            gates = None
            if config.gate_attention:
                if config.merge_heads is None:
                    if not config.dc_gates:
                        gates = layers.Dense(config.num_heads,
                                            activation='sigmoid',
                                            name=f'attention_gates_{tag}',
                                            kernel_regularizer=l2reg)(e)
                    else:
                        gates = layers.Dense(config.num_heads,
                                            name=f'dense_edge_gates_{tag}',
                                            kernel_regularizer=l2reg)(e)
                else:
                    gates = layers.Dense(1,
                                         activation='sigmoid',
                                         name=f'attention_gates_{tag}',
                                         kernel_regularizer=l2reg)(e)
                    gates = layers.Lambda(lambda tensor: tf.squeeze(tensor, axis=-1),
                                          name=f'squeeze_gates_{tag}')(gates)
                # Analysis
                self.analysis.add_analysis(f'attention_gates_{tag}', gates=gates)
            
            e = edge_channel_contrib(tag, e)
            # Analysis
            self.analysis.add_analysis(f'dense_edge_b_{tag}', e=e)
            
            h, e = mha_block(tag, h, e, gates)
            
            e = layers.Dense(config.edge_width, name=f'dense_edge_r_{tag}',
                             kernel_regularizer=l2reg)(e)
            if config.edge_dropout > 0:
                e=layers.Dropout(config.edge_dropout, name=f'drp_edge_{tag}')(e)
            e = layers.Add(name=f'res_edge_{tag}')([e,y])

            if config.add_n_norm:
                e = normlr_edge(name=f'norm_edge_{tag}')(e) 
            
            return h, e


        # FFN
        xtalk_flag = (config.node2edge_xtalk > 0. or 
                      config.edge2node_xtalk > 0.)
        
        def ffnlr1(tag, x, width, normlr):
            y = x
            if not config.add_n_norm:
                x = normlr(name=f'norm_fnn_{tag}')(x)         
            x = layers.Dense(round(width*config.ffn_multiplier),
                             activation=(config.activation 
                                         if not xtalk_flag else None),
                             name=f'fnn_lr1_{tag}',
                             kernel_regularizer=l2reg)(x)
            return x, y
        
        def ffnact(tag, x):
            if xtalk_flag:
                return layers.Activation(config.activation,
                                        name=f'ffn_activ_{tag}')(x)
            else:
                return x
        
        def ffnlr2(tag, x, y, width, normlr, drpr):
            x = layers.Dense(width,
                             name=f'fnn_lr2_{tag}',
                             kernel_regularizer=l2reg)(x)
            if drpr>0:
                x=layers.Dropout(drpr, name=f'drp_fnn_{tag}')(x)
            x=layers.Add(name=f'res_fnn_{tag}')([x,y])

            if config.add_n_norm:
                x = normlr(name=f'norm_fnn_{tag}')(x)
            return x
        
        def channel_xtalk(tag, x_h, x_e):
            node2edge_xtalk = config.node2edge_xtalk
            edge2node_xtalk = config.edge2node_xtalk
            ffn_multiplier = config.ffn_multiplier
            def xtalk_fn(inputs,mask):
                x_h, x_e = inputs
                m_h, _ = mask
                
                x_h_n = None
                if edge2node_xtalk > 0.:
                    nx_s = round(edge2node_xtalk*x_e.shape[-1]/ffn_multiplier)
                    nx_t = x_e.shape[-1] - nx_s*2
                    x_er, x_ec, x_e = tf.split(x_e, [nx_s, nx_s, nx_t], axis=3)
                    
                    m_h = tf.cast(m_h, x_h.dtype)
                    x_er = tf.reduce_sum(x_er * m_h[:,:,None,None], axis=1)
                    x_ec = tf.reduce_sum(x_ec * m_h[:,None,:,None], axis=2)
                    
                    m_h_sum = tf.reduce_sum(m_h, axis=1)[:,None,None]
                    x_h_n = tf.math.divide_no_nan(x_er + x_ec, m_h_sum)
                    
                    x_h_n.set_shape([None,None,nx_s])
                    x_e.set_shape([None,None,None,nx_t])
                    
                x_e_n = None
                if node2edge_xtalk > 0.:
                    nx_s = round(node2edge_xtalk*x_h.shape[-1]/ffn_multiplier)
                    nx_t = x_h.shape[-1] - nx_s*2
                    x_hr, x_hc, x_h = tf.split(x_h, [nx_s, nx_s, nx_t], axis=2)
                    x_e_n = x_hr[:,:,None,:] + x_hc[:,None,:,:]
                    
                    x_e_n.set_shape([None,None,None,nx_s])
                    x_h.set_shape([None,None,nx_t])
                
                if x_h_n is not None:
                    x_h = tf.concat([x_h,x_h_n], axis=-1)
                if x_e_n is not None:
                    x_e = tf.concat([x_e,x_e_n], axis=-1)
                
                return x_h, x_e
            
            def compute_mask(inputs, mask):
                return mask
                 
            if xtalk_flag:
                x_h, x_e = layers.Lambda(xtalk_fn, mask=compute_mask, 
                                         name=f'xtalk_{tag}')([x_h,x_e]) 
            return x_h, x_e
        
        def ffn_block(tag, x_h, x_e):
            tag_h = 'node_' + tag
            x_h, y_h = ffnlr1(tag_h, x_h, config.model_width, normlr_node)
            
            if self.config.edge_channel_type in ['residual', 'constrained']:
                tag_e = 'edge_' + tag
                x_e, y_e = ffnlr1(tag_e, x_e, config.edge_width, normlr_edge)
                
                x_h, x_e = channel_xtalk(tag, x_h, x_e)
                
                x_e = ffnact(tag_e, x_e)
                x_e = ffnlr2(tag_e, x_e, y_e,  config.edge_width, normlr_edge, config.edge_dropout)
                
            x_h = ffnact(tag_h, x_h)
            x_h = ffnlr2(tag_h, x_h, y_h, config.model_width, normlr_node, config.node_dropout)
            return x_h, x_e
        
        
        # Build model
        edge_update_fn_dict = dict(
            none        = edge_update_none,
            constrained = edge_update_residual,
            bias        = edge_update_bias,
            residual    = edge_update_residual,
        )
        edge_update = edge_update_fn_dict[config.edge_channel_type]
        h, e = emb_nodes, emb_edges
        for ii in range(config.model_height):
            ii_tag = f'{ii:0>2d}'
            with layers.namespace(f'layer/{ii}/attention'):
                h, e = edge_update(ii_tag, h, e)
            with layers.namespace(f'layer/{ii}/ffn'):
                h, e = ffn_block(ii_tag, h, e)
        
        if (not config.add_n_norm) and config.do_final_norm:
            with layers.namespace(f'final'):
                h = normlr_node(name='node_norm_final')(h)
                if self.config.edge_channel_type in ['residual', 'constrained']:
                    e = normlr_edge(name='edge_norm_final')(e)

        if not return_all:
            return h, e
        else:
            return h, e, all_node_repr, all_edge_repr
    
    def mlp_out(self, inputs, tag=None):
        config = self.config
        layers = self.tracked_layers

        l2reg = self.l2reg
        
        x = inputs

        for ii,f in enumerate(config.mlp_layers):
            if tag is None:
                lr_name = f'mlp_out_{ii:0>1d}'
            else:
                lr_name = f'mlp_out_{tag}_{ii:0>1d}'
            x = layers.Dense(round(f*config.model_width),
                                activation=config.activation,
                                name=lr_name,
                                kernel_regularizer=l2reg)(x)
        outputs = x                                
        return outputs
    
    def get_node_inputs(self):
        return {}
    
    def get_edge_inputs(self):
        return {}
    
    def get_additional_inputs(self):
        return {}
    
    def create_embedding(self, input_name, input_tensor):
        raise NotImplementedError
    
    def update_across_embeddings(self, node_inputs, edge_inputs,
                                 node_embeddings, edge_embeddings):
        return
    
    def combine_node_embeddings(self, node_embeddings):
        layers = self.tracked_layers
        
        if len(node_embeddings) > 1:
            h = layers.Add(name='node_emb_add')(list(node_embeddings.values()))
        elif len(node_embeddings) == 1:
            h, = list(node_embeddings.values())
        else:
            raise NotImplementedError
        return h
    
    def combine_edge_embeddings(self, edge_embeddings):
        layers = self.tracked_layers
        
        if len(edge_embeddings) > 1:
            e = layers.Add(name='edge_emb_add')(list(edge_embeddings.values()))
        elif len(edge_embeddings) == 1:
            e, = list(edge_embeddings.values())
        else:
            e = None
        
        return e
    
    def get_embeddings(self, node_inputs, edge_inputs):
        layers = self.tracked_layers
        config = self.config

        def get_emb_dict(inputs):
            emb_dict = {}
            for name, inp in inputs.items():
                emb = self.create_embedding(name, inp)
                if emb is not None:
                    emb_dict[name] = emb
            return emb_dict
        
        node_embeddings = get_emb_dict(node_inputs)
        edge_embeddings = get_emb_dict(edge_inputs)
        
        self.update_across_embeddings(node_inputs, edge_inputs,
                                      node_embeddings, edge_embeddings)
            
        h = self.combine_node_embeddings(node_embeddings)
        e = self.combine_edge_embeddings(edge_embeddings)
        return h, e
    
    def get_edge_mask(self, edge_inputs):
        return None
    
    def readout_embeddings(self, h, e):
        raise NotImplementedError
    
    def get_additional_targets(self, node_inputs, edge_inputs):
        return {}
    
    def add_additional_losses(self, additional_targets, h, e,  h_all=None, e_all=None):
        return h, e
    
    def call(self, node_inputs, edge_inputs):
        config = self.config
        layers = self.tracked_layers
        
        with layers.namespace('model/targets'):
            additional_targets = self.get_additional_targets(node_inputs, edge_inputs)
        
        with layers.namespace('model/base'):
            h, e = self.get_embeddings(node_inputs, edge_inputs)
            edge_mask = self.get_edge_mask(edge_inputs)
        
        if config.global_step_layer:
            with layers.namespace('global'):
                h = layers.GlobalStep(name='global_step_tracker')(h)
        
        if not self.config.combine_layer_repr:
            with layers.namespace('model/transform'):
                h, e = self.transform_embeddings(h, e, edge_mask)
            with layers.namespace('model/losses'):
                h, e = self.add_additional_losses(additional_targets, h, e)
            with layers.namespace('model/top'):
                outputs = self.readout_embeddings(h, e)
        else:
            with layers.namespace('model/transform'):
                h, e, h_all, e_all = self.transform_embeddings(h, e, edge_mask, True)
            with layers.namespace('model/losses'):
                h, e = self.add_additional_losses(additional_targets, h, e,  h_all, e_all)
            with layers.namespace('model/top'):
                outputs = self.readout_embeddings(h, e, h_all, e_all)

        return outputs
    
    def get_model(self):
        node_inputs = self.get_node_inputs()
        edge_inputs = self.get_edge_inputs()
        additional_inputs = self.get_additional_inputs()

        inputs = list(node_inputs.values()) + list(edge_inputs.values())\
                        +list(additional_inputs.values())
        outputs = self.call(node_inputs, edge_inputs)

        model = models.Model(inputs, outputs)

        return model
    
    def get_analysis_model(self):
        if not self.analysis.is_analysing():
            raise Exception('No analysis running!')
        node_inputs = self.get_node_inputs()
        edge_inputs = self.get_edge_inputs()
        additional_inputs = self.get_additional_inputs()

        inputs = list(node_inputs.values()) + list(edge_inputs.values())\
                        +list(additional_inputs.values())
        self.call(node_inputs, edge_inputs)
        outputs = self.analysis.get_all_analysis()

        model = models.Model(inputs, outputs)

        return model


