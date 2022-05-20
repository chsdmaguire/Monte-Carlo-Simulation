import pandas as pd
import numpy as np
import sqlalchemy
from pylab import plot, show, grid, xlabel, ylabel
from math import sqrt
from scipy.stats import norm
engine = sqlalchemy.create_engine('postgresql://webadmin:SVR^Ix9u2$GA@51.222.87.240/website')
# ASSUMPTIONS
# ticker = "'SP500'"
# df = pd.read_sql('select distinct * from econ.indice_data where series_id = {} order by time asc'.format(ticker), engine)

# returns = df['value'].pct_change()
# returns = returns.dropna()

# mean = returns.mean()
# td = returns.std()
# print(mean, td)

# currentPortVal = 10000

# numSims = 1000

# def gbmModel(avg, t, std, r):
#     change = (avg * t) + (std * r * (t ** .5)) 
#     return change

# tradingDays = 252
# numYears = 5

# portSims = []

# for i in range(10):
#     randVariable = np.random.random()
#     print(randVariable)

def brownian(x0, n, dt, delta, out=None):
     x0 = np.asarray(x0)
     r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
     if out is None:
        out = np.empty(r.shape)
        np.cumsum(r, axis=-1, out=out)
        out += np.expand_dims(x0, axis=-1)
        return out

# The Wiener process parameter.
delta = 2
# Total time.
T = 10.0
# Number of steps.
N = 500
# Time step size
dt = T/N
# Number of realizations to generate.
m = 20
# Create an empty array to store the realizations.
x = np.empty((m,N+1))
# Initial values of x.
x[:, 0] = 50

res = brownian(x[:,0], N, dt, delta, out=x[:,1:])
print(res)
t = np.linspace(0.0, N*dt, N+1)
for k in range(m):
    plot(t, x[k])
xlabel('t', fontsize=16)
ylabel('x', fontsize=16)
grid(True)
show()