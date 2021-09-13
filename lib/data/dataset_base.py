import tensorflow as tf
import h5py
from pathlib import Path

from .reader import get_meta, get_tokens, RecordReader

class DatasetBase:
    def __init__(self, 
                 dataset_path, 
                 splits          = ['training', 'validation'],
                 shuffle_splits  = ['training'],
                 max_shuffle_len = 10000,
                 prefetch_batch  = True,
                 ):
        self.dataset_path       = dataset_path
        self.splits             = splits         
        self.shuffle_splits     = shuffle_splits 
        self.max_shuffle_len    = max_shuffle_len
        self.prefetch_batch     = prefetch_batch
        
        self.db_file = h5py.File(self.dataset_path, 'r')

        self.record_tokens  = {}
        self.datasets       = {}
        
        self._shuffle_tokens = True
        self._shuffle_db = True
                       
    def __del__(self):
        self.db_file.close()
    
    def get_metadata(self):
        return get_meta(self.db_file, self.get_dataset_name())
    
    def get_dataset_name(self):
        raise NotImplementedError
    def get_record_names(self):
        raise NotImplementedError
    def get_record_keys(self):
        raise NotImplementedError
    def get_record_types(self):
        raise NotImplementedError
    def get_record_shapes(self):
        raise NotImplementedError
    def get_paddings(self):
        raise NotImplementedError
    def get_padded_shapes(self):
        raise NotImplementedError
    def is_chunked(self):
        return False
    
    def _maybe_tuple(self, vals):
        vals = tuple(vals)
        if len(vals) > 1:
            return vals
        elif len(vals) == 1:
            return vals[0]
        elif len(vals) == 0:
            return None
        
    def _ensure_dict(self, *vals):
        rvals = []
        for val in vals:
            if not isinstance(val,dict):
                rvals.append(dict((k,val) for k in self.splits))
            else:
                rvals.append(val)
        return self._maybe_tuple(rvals)

    def load_records(self, split):
        AT = tf.data.experimental.AUTOTUNE

        record_tokens = get_tokens(self.db_file, self.get_dataset_name(), split,
                                   self.is_chunked())
        self.record_tokens[split] = record_tokens
        
        ds_tokens = tf.data.Dataset.from_tensor_slices(record_tokens)
        if (split in self.shuffle_splits) and self._shuffle_tokens:
            ds_tokens = ds_tokens.shuffle(len(record_tokens))
        
        record_reader = RecordReader(self.db_file, 
                                     self.get_record_names(), self.get_record_keys(), 
                                     self.get_record_types(), self.get_record_shapes())
        db_records = ds_tokens.map(record_reader, AT)
        return db_records
    
    def load_split(self, split):
        return self.load_records(split)
    
    def map_data_split(self, split, data):
        return data

    def load_data(self):
        for split in self.splits:
            if split not in self.datasets:
                self.datasets[split] = self.map_data_split(split, 
                                                           self.load_split(split))
        return self._maybe_tuple(self.datasets[s] for s in self.splits)

    def get_batched_split(self, split, batch_size, drop_remainder=False):
        dataset = self.datasets[split]
        if (split in self.shuffle_splits) and self._shuffle_db:
            buffer_size = min(len(self.record_tokens[split]), self.max_shuffle_len)
            dataset = dataset.shuffle(buffer_size)

        all_paddings = self.get_paddings()
        paddings = dict((k,all_paddings[k]) for k in dataset.element_spec)
        all_shapes = self.get_padded_shapes()
        shapes = dict((k,all_shapes[k]) for k in dataset.element_spec)
        return dataset.padded_batch(batch_size, shapes, paddings,
                                    drop_remainder)
    
    def get_batched_data(self, batch_size, drop_remainder=False, map_fns=None):
        batch_size, drop_remainder, map_fns \
                        = self._ensure_dict(batch_size, drop_remainder, map_fns)
        
        self.load_data()
        
        batched_splits = []
        for split in self.splits:
            batched_split = self.get_batched_split(split, batch_size[split],
                                            drop_remainder[split])
            map_fn = map_fns[split]
            if map_fn is not None:
                batched_split = batched_split.map(map_fn)
            if self.prefetch_batch:
                AT = tf.data.experimental.AUTOTUNE
                batched_split = batched_split.prefetch(AT)
            batched_splits.append(batched_split)
        return self._maybe_tuple(batched_splits)
    
    def cache(self, paths=None, clear=False):
        self.load_data()

        if paths is None:
            for split in self.splits:
                self.datasets[split] = self.datasets[split].cache()
            
        else:
            if not isinstance(paths, dict):
                paths = dict((k,Path(paths)/k) for k in self.splits)
            else:
                paths = dict((k,Path(paths[k])) for k in self.splits)
            
            for split in self.splits:
                path = paths[split]
                if path.exists():
                    if clear:
                        for f in path.glob('*'):
                            assert('.index' in f.name or '.data' in f.name)
                            f.unlink()
                else:
                    path.mkdir(parents=True, exist_ok=True)
                self.datasets[split] = self.datasets[split].cache(str(path/split))
        
        return self._maybe_tuple(self.datasets[s] for s in self.splits)
    
    def map(self, functions):
        self.load_data()
        functions = self._ensure_dict(functions)
        
        self.datasets = dict((s, d.map(functions[s])) 
                             for s,d in self.datasets.items() 
                             if functions[s] is not None)
        
        return self._maybe_tuple(self.datasets[s] for s in self.splits)





