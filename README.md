# Privugger-AG: Automatic Attacker Generation for Probabilistic Privacy Risk Analysis

The Master Thesis by: Mathias Oliver Valdbjørn Jørgensen

The tool is made to automatically maximise leakage of a privacy preserving mechanism

Evaluations can be seen in `Evaluation Report.ipynb`, figure about performance can be seen in `Figure Generation Report.ipynb` and finally the comparison with other DFO is found in `DFO.ipynb`

- Report based on tool will be available after the 10th of June 2021.

## How to install the tool
```
git clone 
cd path
pip install -r requirements.txt
```

## How to use the tool
```python
import privuggerag as attacker
from typing import List

#The PPM
def q(a: List[float]) -> float:
    return sum(a)/len(a)
    
#Domain specification
domain = [
    {
        "name": "age", "lower": 10, "upper": 50, "type": "float",
    }
]

#Leakage measurement (will in this case minimise the sum of alice age)
def lm(trace):
    return sum(trace["Alice_age"])

#Run analysis
result = attacker.construct_analysis(q, domain, lm, cores=4) 
print(result)
```

For more exsamples see /Examples/ folder

## Abstract:
In the context of Privacy Risk Analysis, I consider the problem of generating attackers in order to measure the leakage of a program disclosing data. I focus on these programs as black-box functions and on leakage measures of rational numbers. My goal is to create a tool that automatically generates and synthesise attackers such that the found attackers maximises any leakage measures.

The current state of the art for analysing leakage is a method called Privug, which estimates leakage by reinterpreting a regular program, probabilistically. It uses probabilistic distributions to model the data, to be disclosed, and Bayesian Inference to estimate the output. This approach allows information theoretical analysis of the program leakage. However, it suffers significantly from its choice of probabilistic distribution, which has to be selected manually. These probability distributions are the attacker this thesis aims at automatically generating and synthesising.

In this paper, I take advantage of the existing research within Privug to make the foundation of how to measure leakage probabilistically. I show that by automatically scoping the domain of possible distributions, based on the signature of a disclosing program, we can maximise the leakage. The maximisation works on any leakage measure that can be represented as a rational number. The maximisation is done using Bayesian optimisation. I compare Bayesian optimisation, as a leakage maximisation algorithm, to Particle Swarm, BOBYQA and Powell's method. I show that the maximisation converges to the global maximum after few executions, of a first-order differentiable black-box function. I demonstrate the accuracy and usability of the tool on both synthetic and real world libraries for disclosing of data. I show that the tool converges to the analytical maximum on Shannon Entropy, Mutual Information and Bayes Risk. 