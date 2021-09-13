import json
import os
from datetime import datetime

def save_results(dataset_name, model_name, split, metrics, configs, state, parent_dir, 
                 file_name_format='{dataset_name}_{model_name}_results'):
    state = dict( (k,v.numpy().tolist()) for k,v in state.items())
    configs = configs.get_dict()
    timestamp = datetime.now().timestamp()
    new_data = dict(
        timestamp    = timestamp,
        dataset_name = dataset_name,
        model_name   = model_name,
        split        = split,
        metrics      = metrics,
        configs      = configs,
        state        = state,
    )
    
    file_name = file_name_format.format(**configs, **state)+'.json'
    file_path = os.path.join(parent_dir, file_name)
    
    if os.path.exists(file_path):
        with open(file_path,'r') as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    data.append(new_data)
    
    with open(file_path, 'w') as fp:
        json.dump(data, fp, indent='\t')