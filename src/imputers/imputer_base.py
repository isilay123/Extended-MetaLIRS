import torch
import numpy as np


class ImputerBase:

    def __init__(self, name):
        self.name = name
        self._skip_normalization = False # disabled by default

    def skip_normalization(self):
        self._skip_normalization = True
        return self

    def impute(self, df, column_names=None):
        raise Exception(f'Imputer:impute() not properly defined for class "{self.name}"')

    def random_test(self, row=10, col=3, p_miss=0.2, random_seed=None, repeat=1, verbose=True):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        print(f'REPEAT={repeat}')
        for i in range(repeat):
            T = torch.rand(row, col)
            mask = (torch.rand(T.shape) < p_miss).double()
            T_nan = T.clone()
            T_nan[mask.bool()] = np.nan
            if verbose:
                print(f'!Generated random tensor, p_miss={p_miss}\n{T_nan}')
                print(f'!Start Impute with "{self.name}" Imputer')
            column_names = [f'c{i}' for i in range(col)]
            T_res = self.impute(T_nan, column_names=column_names)
            if verbose:
                print(f'!Result\n{T_res}')
        
