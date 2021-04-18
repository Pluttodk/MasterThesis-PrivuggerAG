import scipy as st
import pymc3 as pm

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