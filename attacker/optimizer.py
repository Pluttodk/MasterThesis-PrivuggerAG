import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy as sc

def map_int_to_cont_dist(name, info, domain, shape,rng=None):
    dist = None
    ids = info[0]
    if ids == 0:
        mu, std = info[1:]

        ##Normal
        # normal = pm.Normal, lower=domain["lower"], upper=domain["upper"]
        dist = [lambda siz: sc.stats.norm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 1:
        a,b = info[1:]
        # f_a = lambda b: 2*mu-b
        # b = 1/3*(mu+2*np.sqrt(3)*np.sqrt(std))
        # a = f_a(b)
        # if a > b:
        #     b,a = a,b
        # elif a == b:
        #     b += 1
        dist = [lambda siz: sc.stats.uniform(a, b).rvs(siz,random_state=rng) for _ in range(shape)]
    elif ids == 2:
        #Half normal
        dist = [lambda siz: sc.stats.halfnorm(mu, std).rvs(siz,random_state=rng) for _ in range(shape)]

    #     dist = pm.Gamma(name, mu=abs(mu), sigma=std, shape=shape)
    # if shape == 1:
    #     return dist
    db = np.empty(shape+1, dtype=object)
    if "alice" in domain and isinstance(domain["alice"], sc.stats._distn_infrastructure.rv_frozen):
        db[0] = lambda siz: domain["alice"].rvs(siz)
    else:
        db[0] = lambda siz: sc.stats.uniform(domain["lower"], domain["upper"]-domain["lower"]).rvs(siz, random_state=rng)
    # db[0] = pm.Uniform(f"Alice_{name}", domain["lower"], domain["upper"], shape=1)
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
        dist = [lambda siz: sc.stats.uniform(int(a), int(b)+1).rvs(siz) for _ in range(shape)]
    elif ids == 1:
        dist = [lambda siz: sc.stats.poisson(mu=mu).rvs(siz) for _ in range(shape)]

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

