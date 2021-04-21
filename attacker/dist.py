import scipy as st
import pymc3 as pm


def normal_domain(domain):
    upper_bound_std = (1/12)*(domain["upper"] - domain["lower"])**2
    return [
        {
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0.1, upper_bound_std),
        }
    ],[
    ]

def uniform_domain(domain, pos):
    """
    We add constraints such that the value will scale correctly. 
    One being the following
    1/12*(x)^2 >= 0.1
    Where X is the scale.

    so we get that scale has to be >= sqrt(6/5)
    """
    upper = domain["upper"]
    return [
        {
            "name": domain["name"]+"_mu",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_std",
            "type": "continuous",
            "domain": (0, domain["upper"]),
        }
    ],[
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'-x[:,{pos+1}]+np.sqrt(6/5)' # Variance >= 0.1
        },
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'x[:,{pos+1}]+x[:,{pos}]-{upper}' # lower+scale <= domain[upper]
        },
    ]

class Distributions:
    libraries = {"pymc3", "scipy"}
    dist = []

    def __init__(self, library="scipy"):
        if library not in self.libraries:
            raise Exception("The tool currently only supports scipy and pymc3 distribution")
        self.library = library
    
    def normal(self, mu, std, name=""):
        if self.library == "scipy":
            n = lambda siz: st.stats.norm(mu, std).rvs(siz)
            self.dist.append(n)
        elif self.library == "pymc3":
            n = lambda s: pm.Normal(name, mu, std)
            self.dist.append(n)
    
    def infer(self, sample):
        if self.library == "pymc3":
            with pm.Model() as model:
                x = [ni(sample) for ni in n]
        elif self.library == "scipy":
            x = [ni(sample) for ni in self.dist]
            return x