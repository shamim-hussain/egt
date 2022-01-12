

import tensorflow as tf
from tensorflow.keras import layers

from ..base.track_layers import CustomLayers 
from ..base.xformer_layers.misc import RandomNeg, RandomNegEig
from ..base.graph_layers.virtual_nodes import (VirtualEdgeEmbedding, 
                                               GetVirtualNodes, 
                                               VirtualEdgeEmbedding, VirtualNodeEmbedding)


from ..base.genutil.loss_layers import SparseXEntropy
class AdjMatModel:
    def __init__(self                            ,
                 use_adj         = True          ,
                 include_xpose   = False         ,
                 upto_hop        = 1             ,
                 clip_hops       = True          ,
                 adj_input_name  = 'graph_matrix',
                 max_degree_enc  = 0             ,
                 bidir_degree    = True          ,
                 distance_loss   = 0.            ,
                 distance_target = 8             ,
                 max_diffuse_t   = 0             ,
                 edge_feat_name  = 'fmat'        ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(
            use_adj         = use_adj         ,
            upto_hop        = upto_hop        ,
            clip_hops       = clip_hops       ,
            adj_input_name  = adj_input_name  ,
            max_degree_enc  = max_degree_enc  ,
            bidir_degree    = bidir_degree    ,
            include_xpose   = include_xpose   ,
            distance_loss   = distance_loss   ,
            distance_target = distance_target ,
            max_diffuse_t   = max_diffuse_t   ,
            edge_feat_name  = edge_feat_name  ,
            )
        custom_layers = CustomLayers(SparseXEntropy,) 
        self.tracked_layers.track_module(custom_layers)
    
    def get_edge_inputs(self):
        config = self.config
        layers = self.tracked_layers
        
        inputs = super().get_edge_inputs()

        if config.use_adj:
            inputs.update(
                adj = layers.Input([config.max_length, config.max_length],
                                   name=config.adj_input_name)
            )
        return inputs
    
    def get_additional_targets(self, node_inputs, edge_inputs):
        config = self.config
        layers = self.tracked_layers
        
        additional_targets = super().get_additional_targets(node_inputs, edge_inputs)
        adj_input = edge_inputs['adj']
        
        if config.distance_loss > 0:
                distance_target = config.distance_target
                def add_hops(mat):
                    hops = [mat]
                    hop_mat = mat
                    for _ in range(distance_target-1):
                        hop_mat = tf.matmul(mat, hop_mat)
                        hop_mat = tf.clip_by_value(hop_mat, 0., 1.)
                        hops.append(hop_mat)
                    return tf.cast(tf.math.round(tf.math.add_n(hops)), tf.int32)
                additional_targets['distance'] = layers.Lambda(add_hops, name='adj_add_hops')(adj_input)
        return additional_targets
    
    def add_additional_losses(self, additional_targets, h, e,  h_all=None, e_all=None):
        config = self.config
        layers = self.tracked_layers
        
        if config.distance_loss > 0:
            e = self.mlp_out(e, tag='dist_targ')
            e = layers.Dense(config.distance_target+1, name='distance_target',
                                kernel_regularizer=self.l2reg)(e)
            
            distance_target = additional_targets['distance']
            distance_target, e, h = layers.SparseXEntropy(config.distance_loss, 
                                    from_logits=True,
                                    mask_zero=True, 
                                    metric_name='distance_loss',
                                    reduction='sum',
                                    name='distance_loss_layer')([distance_target, e, h])
        return h, e
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'adj':
            if config.upto_hop == 1:
                x = layers.Lambda(lambda v: v[...,None], name='adj_expand_dim')(x)
            elif config.upto_hop > 1:
                upto_hop = config.upto_hop
                clip_hops = config.clip_hops
                def stack_hops(mat):
                    hops = [mat]
                    hop_mat = mat
                    for _ in range(upto_hop-1):
                        hop_mat = tf.matmul(mat, hop_mat)
                        if clip_hops:
                            hop_mat = tf.clip_by_value(hop_mat, 0., 1.)
                        hops.append(hop_mat)
                    return tf.stack(hops, axis=-1)
                x = layers.Lambda(stack_hops, name='adj_stack_hops')(x)
            else:
                raise ValueError
            
            if self.config.include_xpose:
                x = layers.Lambda(lambda v: tf.concat([v, tf.transpose(v, perm=[0,2,1,3])], axis=-1),
                                  name='adj_include_transpose')(x)
            
            x = layers.Dense(config.edge_width, name='adj_emb',
                             kernel_regularizer=self.l2reg)(x)
        else:
            x = super().create_embedding(name, x)
        
        return x
    
    def get_edge_mask(self, edge_inputs):
        config = self.config
        layers = self.tracked_layers

        if config.edge_channel_type == 'constrained':
            adj_mat = edge_inputs['adj']
            nh = config.num_heads
            edge_mask = layers.Lambda(lambda v: tf.tile(v[...,None],[1,1,1,nh]),
                                      name='adj_expand_mask')(adj_mat)
            return edge_mask
        else:
            return super().get_edge_mask(edge_inputs)

    def update_across_embeddings(self, node_inputs, edge_inputs,
                                 node_embeddings, edge_embeddings):
        config = self.config
        layers = self.tracked_layers
        
        if config.use_adj and config.max_degree_enc > 0:
            max_degree = config.max_degree_enc
            def bi_degree_oh(adj):
                in_deg = tf.cast(tf.minimum(tf.reduce_sum(adj, axis=1), 
                                            max_degree), tf.int32)
                out_deg = tf.cast(tf.minimum(tf.reduce_sum(adj, axis=2), 
                                            max_degree), tf.int32)
                
                in_deg_oh = tf.one_hot(in_deg, max_degree+1, dtype=tf.float32)
                out_deg_oh = tf.one_hot(out_deg, max_degree+1, dtype=tf.float32)
                
                node_oh = tf.concat([in_deg_oh, out_deg_oh], axis=-1)
                return node_oh
            
            def uni_degree_oh(adj):
                in_deg = tf.cast(tf.minimum(tf.reduce_sum(adj, axis=1), 
                                            max_degree), tf.int32)
                in_deg_oh = tf.one_hot(in_deg, max_degree+1, dtype=tf.float32)
                return in_deg_oh
            
            x = edge_inputs['adj']
            x = layers.Lambda(bi_degree_oh if config.bidir_degree else uni_degree_oh, 
                              name='degree_enc_oh')(x)
            x = layers.Dense(config.model_width, name='degree_emb',
                             kernel_initializer='uniform',
                             kernel_regularizer=self.l2reg)(x)
            node_embeddings['deg'] = x
        
        
        if config.use_adj and config.max_diffuse_t > 0:
            max_diffuse_t = config.max_diffuse_t
            edge_feat_name  = config.edge_feat_name
            def cat_diffusions(inputs, mask):
                e, A = inputs
                mask = mask[0]
                
                A = tf.math.divide_no_nan(A, tf.reduce_sum(A, axis=1, keepdims=True))
                e = e * tf.cast(tf.expand_dims(mask, axis=-1), e.dtype)
                
                eds = []
                ed = e
                for _ in range(max_diffuse_t):
                    ed = tf.einsum('bij,bjkl->bikl', A, ed)
                    eds.append(ed)
                
                edf = tf.concat(eds, axis=-1)
                return edf
            
            def compute_mask(inputs, mask):
                return mask[0]
            
            feat_emb = edge_embeddings[edge_feat_name]
            adj_mat = edge_inputs['adj']
            x = layers.Lambda(cat_diffusions, mask=compute_mask,
                              name='stack_edge_diffusions')([feat_emb, adj_mat])
            x = layers.Dense(config.edge_width, name='diffusion_emb',
                                kernel_regularizer=self.l2reg)(x)
            edge_embeddings['diffusion'] = x
        
        
        return super().update_across_embeddings(node_inputs, edge_inputs,
                                                node_embeddings, edge_embeddings)    
    
class VNModel:
    def __init__(self                   ,
                 num_virtual_nodes  = 1 ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(
            num_virtual_nodes  = num_virtual_nodes  ,
        )
        
        custom_layers = CustomLayers(VirtualNodeEmbedding,
                                     VirtualEdgeEmbedding,
                                     GetVirtualNodes) 
        self.tracked_layers.track_module(custom_layers)
    
    def combine_node_embeddings(self, node_embeddings):
        layers = self.tracked_layers
        config = self.config
        
        h = super().combine_node_embeddings(node_embeddings)
        if config.num_virtual_nodes > 0:
            h = layers.VirtualNodeEmbedding(num_nodes=config.num_virtual_nodes,
                                            name='virtual_node_embedding')(h)
        return h
    
    def combine_edge_embeddings(self, edge_embeddings):
        layers = self.tracked_layers
        config = self.config
        
        e = super().combine_edge_embeddings(edge_embeddings)
        if e is not None and config.num_virtual_nodes > 0:
            e = layers.VirtualEdgeEmbedding(num_nodes=config.num_virtual_nodes,
                                            name='virtual_edge_embedding')(e)
        
        return e
    
    def get_edge_mask(self, edge_inputs):
        config = self.config
        layers = self.tracked_layers
        
        edge_mask = super().get_edge_mask(edge_inputs)

        if edge_mask is not None:
            num_nodes = config.num_virtual_nodes
            def expand_mask(e_mask):
                bshape_d, eshape1_d, eshape2_d, nh_d = tf.unstack(tf.shape(e_mask))
                row_true = tf.ones([bshape_d, num_nodes, eshape2_d, nh_d], dtype=e_mask.dtype)
                col_true = tf.ones([bshape_d, eshape1_d+num_nodes, num_nodes, nh_d], dtype=e_mask.dtype)
                
                e_mask = tf.concat([row_true, e_mask], axis=1)
                e_mask = tf.concat([col_true, e_mask], axis=2)
                return e_mask
            
            edge_mask = layers.Lambda(expand_mask,
                                      name='virtual_node_expand_mask')(edge_mask)
        
        return edge_mask
        
    def add_additional_losses(self, additional_targets, h, e,  h_all=None, e_all=None):
        config = self.config
        layers = self.tracked_layers
        
        num_nodes = config.num_virtual_nodes
        def crop_ec(inputs,mask=None):
            return inputs[:,num_nodes:,num_nodes:,:]
        def crop_mask(inputs, mask):
            return mask[:,num_nodes:,num_nodes:]
        
        e = layers.Lambda(crop_ec,mask=crop_mask,name='crop_edge_channels')(e)
        return super().add_additional_losses(additional_targets, h, e,  h_all, e_all)      


class SVDFeatModel:
    def __init__(self                                    ,
                 use_svd            = False              ,
                 num_svd_features   = 256                ,
                 sel_svd_features   = 128                ,
                 random_neg         = False              ,
                 transform_svd      = False              ,
                 svdf_input_name    = 'singular_vectors' ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(
            num_svd_features   = num_svd_features   ,
            sel_svd_features   = sel_svd_features   ,
            use_svd            = use_svd            ,
            transform_svd      = transform_svd      ,
            random_neg         = random_neg         ,
            svdf_input_name    = svdf_input_name    ,
        )
        
        custom_layers = CustomLayers(RandomNeg) 
        self.tracked_layers.track_module(custom_layers)
    

    def get_node_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_node_inputs()

        if config.use_svd:
            inputs.update(
                svdf  = layers.Input([config.max_length, 
                                      config.num_svd_features, 2],
                                      name=config.svdf_input_name)
            )
        return inputs
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'svdf':
            mw     = config.model_width
            sf     = config.sel_svd_features
            tsvd_f = config.transform_svd
            def process_svd(v):
                pad_len = max(0, mw//2 - sf)
                v = v[:,:,:sf,:]
                if not tsvd_f:
                    v = tf.pad(v, [(0,0),(0,0),(0,pad_len),(0,0)])
                return v
            x = layers.Lambda(process_svd , name='svd_process')(x)
            
            if config.random_neg:
                x = layers.RandomNeg(name='random_neg')(x)
            
            x = layers.Lambda(lambda v: tf.concat(tf.unstack(v, num=2, axis=-1), axis=-1),
                              name='svd_flatten')(x)
            
            if config.transform_svd:
                x = layers.Dense(config.model_width, name='svd_emb',
                                  kernel_regularizer=self.l2reg)(x)
        else:
            x = super().create_embedding(name, x)
        return x


class EigFeatModel:
    def __init__(self                                    ,
                 use_eig            = False              ,
                 num_eig_features   = 40                 ,
                 sel_eig_features   = 20                 ,
                 random_neg         = False              ,
                 transform_eig      = False              ,
                 eigf_input_name    = 'eigen_vectors'    ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.config.__dict__.update(
            num_eig_features   = num_eig_features   ,
            sel_eig_features   = sel_eig_features   ,
            use_eig            = use_eig            ,
            transform_eig      = transform_eig      ,
            random_neg         = random_neg         ,
            eigf_input_name    = eigf_input_name    ,
        )
        
        custom_layers = CustomLayers(RandomNegEig) 
        self.tracked_layers.track_module(custom_layers)
    

    def get_node_inputs(self):
        config = self.config
        layers = self.tracked_layers

        inputs = super().get_node_inputs()

        if config.use_eig:
            inputs.update(
                eigf  = layers.Input([config.max_length, 
                                      config.num_eig_features],
                                      name=config.eigf_input_name)
            )
        return inputs
    
    def create_embedding(self, name, x):
        config = self.config
        layers = self.tracked_layers

        if name == 'eigf':
            mw     = config.model_width
            sf     = config.sel_eig_features
            teig_f = config.transform_eig
            def process_eig(v):
                pad_len = max(0, mw - sf)
                v = v[:,:,:sf]
                if not teig_f:
                    v = tf.pad(v, [(0,0),(0,0),(0,pad_len)])
                return v
            x = layers.Lambda(process_eig , name='eig_pad')(x)
            
            if config.random_neg:
                x = layers.RandomNegEig(name='random_neg')(x)
                    
            if config.transform_eig:
                x = layers.Dense(config.model_width, name='eig_emb',
                                  kernel_regularizer=self.l2reg)(x)
        else:
            x = super().create_embedding(name, x)
        return x