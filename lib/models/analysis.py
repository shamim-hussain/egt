
from contextlib import contextmanager

class Analysis:
    def __init__(self):                              
        self._analysis = None
        self._analysis_out = None

    @contextmanager
    def __call__(self, analysis_dict=None, **kwargs):
        if self.is_analysing():
            raise Exception('Already running an analysis!')
        try:
            self._analysis = kwargs
            if analysis_dict is not None:
                self._analysis.update(analysis_dict)
            self._analysis_out = dict((name,dict()) for name in self._analysis)
            yield self._analysis_out
        finally:
            self._analysis = None
            self._analysis_out = None
    
    def is_analysing(self, name=None):
        if name is None:
            return (self._analysis is not None)
        else:
            return (self._analysis is not None
                    and name in self._analysis)
    
    def get_analysis(self):
        return self._analysis_out
    
    def add_analysis(self, name, analyses_dict=None, **analyses):
        if analyses_dict is not None:
            analyses.update(analyses_dict)
        if self.is_analysing(name):
            for k in self._analysis[name]:
                self._analysis_out[name][k] = analyses[k]
    
    def get_all_analysis(self):
        return list(a for d in self._analysis_out.values() 
                              for a in d.values())


