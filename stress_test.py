import numpy as np

def simulate_stress(portfolio_pd, shock=0.1):
    stressed_pd = portfolio_pd * (1 + shock)
    expected_loss = stressed_pd.mean()
    return expected_loss
