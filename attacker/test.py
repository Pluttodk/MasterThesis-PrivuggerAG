def f(age : List[float]) -> float:
    return sum(age)/len(age)

def q_mutual_info(trace):
    I = mutual_info_regression(trace["Alice_age"][:,0].reshape(-1,1), trace["out"][:,0])[0]
    return -I

domain = [
    {
        "name": "age", 
        "lower": 0, 
        "upper": 100,
        "type": "float"
    },
]
# domain_to_dist(domain)
bp = construct_analysis(f, domain, q_mutual_info)
bp.plot_convergence()
# run_opt(analys, domain).plot_convergence()
plt.savefig("AverageAgeExample")