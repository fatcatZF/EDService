import joblib 
import torch 
import numpy as np 

from abc import ABC, abstractmethod


class DriftDetector(ABC):
    @abstractmethod
    def load(self, filepath):
        """Load a trained model from a file"""
        raise NotImplementedError()
    
    @abstractmethod
    def compute_drift_score(self, st: list, at: list, stplus1:list, 
                            rtplus1:list|None = None):
        """Compute a drift score given input x"""
        raise NotImplementedError()
    

    

class LOFDetector(DriftDetector):
    def __init__(self, filepath):
        self.load(filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)

    def compute_drift_score(self, st: list, at: list|int, 
                            stplus1: list,
                            rtplus1: float|None = None):
        
        st = np.array(st)
        stplus1 = np.array(stplus1)
        if type(at) == int:
            at = np.array([at])
        else:
            at = np.array(at)
        
        x = np.concatenate([st, stplus1-st, at]).reshape(1, -1)
        
        return -self.model.decision_function(x)[0]
    




class EDSVMDetector(DriftDetector):
    def __init__(self, filepath):
        self.load(filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)

    def compute_drift_score(self, st:list, at:list|int, 
                            stplus1:list, 
                            rtplus1:float|None=None):
        
        st = np.array(st)
        stplus1 = np.array(stplus1)
        if type(at) == int:
            at = np.array([at])
        else:
            at = np.array(at)
        
        x = np.concatenate([st, stplus1-st, at]).reshape(1, -1)

        return -self.model.decision_function(x)[0]
    



    




