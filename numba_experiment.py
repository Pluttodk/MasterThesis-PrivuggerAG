from numba import jit
from scipy.stats import norm
import numpy as np

@jit(nopython=True)
def normal_mean(nsamples):
    x = np.random.normal(0,10, size=nsamples)
    return np.mean(x)

def normal_mean_without(nsamples):
    x = np.random.normal(0,10, size=nsamples)
    return np.mean(x)

print(normal_mean(10_000_000))