from matplotlib import pyplot as plt
import numpy as np
from scipy import stats as st
import time
import pymc3 as pm
import pandas as pd
import seaborn as sns

samples = [100,1000,10_000,100_000, 1_000_000]


def mean_estimate(samples):
    def py_normal(s):
        with pm.Model() as model:
            t = pm.Normal("no", 0, 10)
            trace = pm.sample(s, cores=1, return_inferencedata=False)
        return trace["no"]

    numpy = [abs(np.mean(np.random.normal(0,10,s))) for s in samples]

    scipy = [abs(np.mean(st.norm(0,10).rvs(s))) for s in samples]

    py = [abs(np.mean(py_normal(s))) for s in samples]


    data_set = {
        "sample size": samples*3,
        "library": ["numpy"]*len(samples)+["scipy"]*len(samples)+["pymc3"]*len(samples),
        "Estimate of mu": numpy+scipy+py
    }
    df = pd.DataFrame(data_set, columns=data_set.keys())
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="sample size", y="Estimate of mu", hue="library", data=df)
    plt.title(r"Estimate of mu from a $\mathbb{N}(\mu=0, \sigma=10)$")
    plt.show()

mean_estimate(samples[:2])
