import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

CONT_DIST = 4
DISC_DIST = 2

def map_int_to_cont_dist(name, info, domain, shape):
    ids, mu, std = info
    dist = None
    if ids == 0:
        ##Normal
        dist = pm.Normal(name, mu, std, shape=shape)
    elif ids == 1:
        #Uniform
        f_a = lambda b: 2*mu-b
        b = 1/3*(mu+2*np.sqrt(3)*np.sqrt(std))
        a = f_a(b)
        if a > b:
            b,a = a,b
        elif a == b:
            b += 1
        dist = pm.Uniform(name, a, b, shape=shape)
    elif ids == 2:
        #Half Normal
        dist = pm.HalfNormal(name, std, shape=shape)
    elif ids == 3:
        #Gamma
        dist = pm.Gamma(name, mu=abs(mu), sigma=std, shape=shape)
    if shape == 1:
        return dist
    db = np.emtpy(shape+1, dtype=object)
    db[0] = pm.Uniform("Alice", domain["lower"], domain["upper"])
    for i in range(shape):
        db[i+1] = dist[i]
    return db

def map_int_to_discrete_dist(name, info, domain, shape):
    ids, mu, std = info
    dist = None
    if ids == 0:
        f_a = lambda b: 2*mu-b
        b = 1/3*(mu+2*np.sqrt(3)*np.sqrt(std))
        a = f_a(b)
        if a > b:
            b,a = a,b
        elif a == b:
            b += 1
        dist = pm.DiscreteUniform(name, int(a), int(b), shape=shape)
    elif ids == 1:
        mu = abs(mu)
        dist = pm.Poisson(name, mu=mu, shape=shape)
    if shape == 1:
        return dist
    db = np.emtpy(shape+1, dtype=object)
    db[0] = pm.Uniform("Alice", domain["lower"], domain["upper"])
    for i in range(shape):
        db[i+1] = dist[i]
    return db

def visualise_model(model, x):
    plt.rcParams["figure.figsize"] = (20,8)
    for rv in model.free_RVs:
        plt.plot(x, np.exp(rv.distribution.logp_nojac(x).eval()), label=rv.__str__())
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()

