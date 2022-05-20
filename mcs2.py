import pandas as pd
import numpy as np
import sqlalchemy
from pylab import plot, show, grid, xlabel, ylabel
from math import sqrt
from scipy.stats import norm

def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    if out is None:
        out = np.empty(r.shape)
        np.cumsum(r, axis=-1, out=out)
        out += np.expand_dims(x0, axis=-1)
        return out

pv = 1000
delta = 2
T = 10.0
N = 3
dt = T/N
m = 2
pvArray = np.empty((m,N+1))

pvArray[:, 0] = pv

res = brownian(pvArray[:, 0], N, dt, delta, out=pvArray[:,1:])
# print(res)

print(pvArray)
print('--------------------------------------')
print(pvArray[:,1:])