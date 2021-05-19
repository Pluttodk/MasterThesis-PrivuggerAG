import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy as sc

def map_int_to_cont_dist(name, info, domain, shape,rng=None):
    dist = None
    ids = info[0]
    
    if ids == 0:
        #Normal

        mu, std = info[1:]
        dist = [lambda siz: sc.stats.norm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 1:
        #UNIFORM
        
        a,b = info[1:]
        dist = [lambda siz: sc.stats.uniform(a, b).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 2:
        #Half normal
        mu, std = info[1:]
        dist = [lambda siz: sc.stats.halfnorm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]

    #     dist = pm.Gamma(name, mu=abs(mu), sigma=std, shape=shape)
    if shape == 1:
        return dist
    db = np.empty(shape+1, dtype=object)
    if "alice" in domain and isinstance(domain["alice"], sc.stats._distn_infrastructure.rv_frozen):
        db[0] = lambda siz: domain["alice"].rvs(siz, random_state=rng)
    else:
        db[0] = lambda siz: sc.stats.uniform(domain["lower"], domain["upper"]-domain["lower"]).rvs(siz, random_state=rng)
    for i in range(shape):
        db[i+1] = dist[i]
    return db

def map_int_to_discrete_dist(name, info, domain, shape):
    ids = info[0]
    dist = None
    if ids == 1:
        a,b = info[1:]
        dist = [lambda siz: sc.stats.randint(a, a+b).rvs(siz) for _ in range(shape)]
    elif ids == 0:
        mu,loc = info[1:]
        dist = [lambda siz: sc.stats.poisson(mu=mu,loc=loc).rvs(siz) for _ in range(shape)]

    if shape == 1:
        return dist
    db = np.empty(shape+1, dtype=object)
    db[0] = lambda siz: sc.stats.randint(domain["lower"], domain["upper"]+1).rvs(siz)
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

