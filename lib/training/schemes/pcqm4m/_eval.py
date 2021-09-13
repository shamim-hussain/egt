
from lib.base.dotdict import HDict
# from lib.training.schemes.evaluation import save_results
import os

class PCQM4MEval:
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            prediction_bmult = HDict.L('c: max((1024 if c.distributed else 128)'+
                                          '//c.batch_size, 1)'),
        )
        return config_dict
    
    def do_evaluations(self):
        self.eval_flag = True
        self.prepare_for_test()
        
        os.makedirs(self.config.predictions_path, exist_ok=True)
        for split in ['trainset', 'valset']:
            print('='*40)
            print(f'Evaluation on {split}.')
            self.do_evaluations_on_split(split)
            print()
    
    def do_evaluations_on_split(self,split):
        loss, mae, *_ = self.model.evaluate(getattr(self,split))
        print(f'{split} MAE = {mae:0.5f}')
        
        save_path = os.path.join(self.config.predictions_path,f'{split}_evals.txt')
        with open(save_path, 'a') as fl:
            print(f'{split} MAE = {mae:0.5f}', file=fl)
        
        # save_results(
        #     dataset_name = self.config.dataset_name,
        #     model_name   = self.config.model_name,
        #     split        = split,
        #     metrics      = dict(
        #         mae          = mae.tolist(),
        #         loss         = loss.tolist(),
        #     ),
        #     configs      =self.config,
        #     state        =self.state,
        #     parent_dir   =self.config.predictions_path,
        # )
