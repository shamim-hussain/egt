
import os
# from lib.training.schemes.evaluation import save_results


class ZINCEval:    
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

