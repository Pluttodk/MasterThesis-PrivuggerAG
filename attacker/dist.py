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
            "name": domain["name"]+"_lower",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (domain["lower"], domain["upper"]) if  domain["type"] == "float" else tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_spread",
            "type": "continuous" if domain["type"] == "float" else "discrete",
            "domain": (0, domain["upper"]-domain["lower"]) if domain["type"] == "float" else tuple(range(0, domain["upper"]-domain["lower"])),
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

def poisson_domain(domain, pos):
    """
    We add constraints such that the value will scale correctly. 
    One being the following
    1/12*(x)^2 >= 0.1
    Where X is the scale.

    so we get that scale has to be >= sqrt(6/5)
    """
    upper_bound_std = domain["lower"]+(1/12)*(domain["upper"] - domain["lower"])**2
    upper = domain["upper"]
    return [
        {
            "name": domain["name"]+"_lambda",
            "type": "discrete",
            "domain": tuple(range(int(domain["lower"]), int(domain["upper"]))),
        },
        {
            "name": domain["name"]+"_loc",
            "type": "discrete",
            "domain": tuple(range(int(domain["lower"]), int(domain["upper"]))),
        }
    ],[
        {
            'name': domain["name"]+"_constr2", 
            'constraint': f'x[:,{pos+1}]+x[:,{pos}]-{upper}' # lower+scale <= domain[upper]
        },
    ]

def half_normal_domain(domain, pos):
    upper_bound_std = (1/12)*(domain["upper"] - domain["lower"])**2
    upper = domain["upper"]
    return [{
        "name": domain["name"]+"_mu",
        "type": "continuous",
        "domain": (domain["lower"], domain["upper"]),
    },{
        "name": domain["name"]+"_std",
        "type": "continuous",
        "domain": (0.1, upper_bound_std)
    }], [{
            'name': domain["name"]+"_constr1", 
            'constraint': f'2*x[:,{pos+1}]+x[:,{pos}]-{upper}' # Variance >= 0.1
        },
    ]