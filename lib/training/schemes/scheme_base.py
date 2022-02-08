
from lib.training.training_base import TrainingBase
from lib.base.dotdict import HDict
from lib.base.genutil.warmup import WarmUpAndCosine

class BaseDCModelScheme(TrainingBase):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
                model_name         = 'dc',
                dataset_name       = 'dataset',
                dataset_path       = HDict.L('c:f"datasets/{c.dataset_name.upper()}/{c.dataset_name.upper()}.h5"'),
                cache_dir          = HDict.L('c:f"data_cache/{c.dataset_name.upper()}/data"'),
                save_path          = HDict.L('c:path.join(f"models/{c.dataset_name.lower()}",c.model_name)'),
                model_width        = 48        ,
                model_height       = 4         ,
                edge_width         = 48        ,
                num_heads          = 8         ,
                gate_attention     = True      ,
                scale_degree       = False     ,
                l2_reg             = 0         ,
                dropout            = 0         ,
                attn_dropout       = 0.0       ,
                edge_dropout       = None      ,
                mlp_layers         = [.5, .25] ,
                edge_activation    = None      ,
                edge_channel_type  = 'residual',
                combine_layer_repr = False     ,
                max_shuffle_len    = 10000     ,
                ffn_multiplier     = 2.        ,
                warmup_steps       = 0         ,
                total_steps        = None      ,
                random_mask_prob   = 0.       ,
            )
        return config_dict
    
    def get_model_config(self):
        config = self.config
        model_config, model_class = super().get_model_config()
        model_config.update(
            model_width        = config.model_width        ,
            edge_width         = config.edge_width         ,
            num_heads          = config.num_heads          ,
            gate_attention     = config.gate_attention     ,
            scale_degree       = config.scale_degree       ,
            random_mask_prob   = config.random_mask_prob   ,
            attn_dropout       = config.attn_dropout       ,
            model_height       = config.model_height       ,
            l2_reg             = config.l2_reg             ,
            node_dropout       = config.dropout            ,
            edge_dropout       = (config.dropout
                                  if config.edge_dropout is None 
                                  else config.edge_dropout),
            mlp_layers         = config.mlp_layers         ,
            edge_channel_type  = config.edge_channel_type  ,
            edge_activation    = config.edge_activation    ,
            ffn_multiplier     = config.ffn_multiplier     ,
            global_step_layer  = True                      ,
        )
        return model_config, model_class
    
    def get_dataset_config(self, splits=['training','validation']):
        config = self.config
        dataset_config, dataset_class = super().get_dataset_config()
        dataset_config.update(dataset_path    = config.dataset_path    ,
                              max_length      = None                   ,
                              max_shuffle_len = config.max_shuffle_len ,
                              )
        return dataset_config, dataset_class
    
    def get_callbacks(self):
        cbacks = super().get_callbacks()
        warmup_steps = self.config.warmup_steps
        if warmup_steps>0:
            wu_cback = WarmUpAndCosine(warmup_steps=warmup_steps,
                                       max_lr = self.config.initial_lr,
                                       total_steps = self.config.total_steps)
            cbacks.append(wu_cback)
        
        return cbacks    


class BaseAdjModelScheme(BaseDCModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            model_name         = 'dc_mat',
            cache_dir          = HDict.L('c:f"data_cache/{c.dataset_name.upper()}/mat"'),
            upto_hop           = 1,
            distance_loss      = 0.,
            distance_target    = 8,
        )
        return config_dict
    
    def get_excluded_features(self):
        excluded_feats = super().get_excluded_features()
        excluded_feats.extend(['record_name', 'num_nodes'])
        return excluded_feats
    
    def get_model_config(self):
        config = self.config
        
        model_config, model_class = super().get_model_config()
        model_config.update(
            upto_hop        = config.upto_hop        ,
            distance_loss   = config.distance_loss   ,
            distance_target = config.distance_target ,
        )
        return model_config, model_class


class BaseSVDModelScheme(BaseAdjModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            model_name         = 'dc_svd',
            cache_dir          = HDict.L('c:f"data_cache/{c.dataset_name.upper()}/svd_{c.num_svd_features}"'),
            num_svd_features   = 16,
            sel_svd_features   = 8, 
            use_svd            = True,
            random_neg         = True,
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        config = self.config
        dataset_config, dataset_class = super().get_dataset_config()
        dataset_config.update(return_mat   = True                    ,
                              normalize    = False                   ,
                              num_features = config.num_svd_features ,
                              norm_for_svd = False                   ,
                              )
        return dataset_config, dataset_class
    
    def get_excluded_features(self):
        excluded_feats = super().get_excluded_features()
        if not self.config.use_svd:
            excluded_feats.append('singular_vectors')
        return excluded_feats
    
    def get_model_config(self):
        config = self.config
        
        model_config, model_class = super().get_model_config()
        model_config.update(
            use_svd            = config.use_svd             ,
            transform_svd      = True      ,
            random_neg         = config.random_neg          ,
            num_svd_features   = config.num_svd_features    ,
            sel_svd_features   = config.sel_svd_features    ,
        )
        return model_config, model_class


class BaseEigModelScheme(BaseAdjModelScheme):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            model_name         = 'dc_eig',
            cache_dir          = HDict.L('c:f"data_cache/{c.dataset_name.upper()}/eig_{c.num_eig_features}"'),
            num_eig_features   = 20,
            sel_eig_features   = 8, 
            use_eig            = True,
        )
        return config_dict
    
    def get_dataset_config(self, splits=['training','validation']):
        config = self.config
        dataset_config, dataset_class = super().get_dataset_config()
        dataset_config.update(num_features = config.num_eig_features)
        return dataset_config, dataset_class
    
    def get_excluded_features(self):
        excluded_feats = super().get_excluded_features()
        if not self.config.use_eig:
            excluded_feats.append('eigen_vectors')
        return excluded_feats
    
    def get_model_config(self):
        config = self.config
        
        model_config, model_class = super().get_model_config()
        model_config.update(
            use_eig            = config.use_eig             ,
            transform_eig      = False                      ,
            random_neg         = True                       ,
            num_eig_features   = config.num_eig_features    ,
            sel_eig_features   = config.sel_eig_features    ,
        )
        return model_config, model_class