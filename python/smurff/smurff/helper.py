import pandas as pd
import numpy as np
import math

from .wrapper import NoiseConfig, DataConfig

def make_dataconfig(data, noise = None, is_scarce = None, pos = None):
    ret = DataConfig()
    if is_scarce is not None:
        ret.setData(data, is_scarce)
    else:
        ret.setData(data)

    if noise is not None:
        ret.setNoiseConfig(noise)

    if pos is not None:
        ret.setPos(pod)

    return ret

class SparseTensor:
    """Wrapper around a pandas DataFrame to represent a sparse tensor

       The DataFrame should have N index columns (int type) and 1 value column (float type)
       N is the dimensionality of the tensor

       You can also specify the shape of the tensor. If you don't it is detected automatically.
    """
       
    def __init__(self, data, shape = None):
        if type(data) == SparseTensor:
            self.data = data.data
            self.nnz = data.nnz

            if shape is not None:
                self.shape = shape
            else:
                self.shape = data.shape
        elif type(data) == pd.DataFrame:
            self.data = data
            self.nnz = len(data.index)

            idx_column_names = list(filter(lambda c: data[c].dtype==np.int64 or data[c].dtype==np.int32, data.columns))
            val_column_names = list(filter(lambda c: data[c].dtype==np.float32 or data[c].dtype==np.float64, data.columns))


            if len(val_column_names) != 1:
                error_msg = "tensor has {} float columns but must have exactly 1 value column.".format(len(val_column_names))
                raise ValueError(error_msg)

            if shape is not None:
                self.shape = shape
            else:
                self.shape = [data[c].max() + 1 for c in idx_column_names]
        else:
            error_msg = "Unsupported sparse tensor data type: {}".format(data)
            raise ValueError(error_msg)

        self.ndim = len(self.shape)

class FixedNoise(NoiseConfig):
    def __init__(self, precision = 5.0): 
        NoiseConfig.__init__(self, "fixed", precision)

class SampledNoise(NoiseConfig):
    def __init__(self, precision = 5.0): 
        NoiseConfig.__init__(self, "sampled", precision)

class AdaptiveNoise(NoiseConfig):
    def __init__(self, sn_init = 5.0, sn_max = 10.0): 
        NoiseConfig.__init__(self, "adaptive", sn_init = sn_init, sn_max = sn_max)

class ProbitNoise(NoiseConfig):
    def __init__(self, threshold = 0.): 
        NoiseConfig.__init__(self, "probit", threshold = threshold)