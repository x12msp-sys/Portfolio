import numpy as np, pandas as pd
from scipy.stats import poisson

def predict_match(mu_home=1.6, mu_away=1.2):
    # Basic Poisson scoring probabilities
    max_goals = 6
    probs = np.outer(poisson.pmf(np.arange(0, max_goals+1), mu_home),
                     poisson.pmf(np.arange(0, max_goals+1), mu_away))
    p_home = probs[np.triu_indices_from(probs, k=1)].sum()
    p_draw = np.trace(probs)
    p_away = probs[np.tril_indices_from(probs, k=-1)].sum()
    return {"1": float(p_home), "X": float(p_draw), "2": float(p_away)}

def main():
    p = predict_match()
    print("Win/Draw/Loss probabilities:", p)

if __name__ == "__main__":
    main()
