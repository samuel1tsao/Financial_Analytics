import math
from scipy.stats import norm

# Risk free rate approx (could fetch dynamically but 0.04 is a good fixed safe assumption for recent years)
DEFAULT_RISK_FREE_RATE = 0.04

def calculate_black_scholes_greeks(S: float, K: float, T: float, sigma: float, r: float = DEFAULT_RISK_FREE_RATE):
    """
    Calculate theoretical Black-Scholes Greeks for a Call option.
    S: Current Stock Price
    K: Strike Price
    T: Time to Expiration (in years, e.g., 30/365)
    sigma: Annualized Volatility (e.g., 0.20 for 20%)
    r: Risk-free interest rate (e.g., 0.04 for 4%)
    """
    # Defensive checks to avoid division by zero
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }

    d1 = (math.log(S / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # N is the standard normal cumulative distribution function
    # N_prime is the standard normal probability density function
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_prime_d1 = norm.pdf(d1)

    delta = N_d1
    gamma = N_prime_d1 / (S * sigma * math.sqrt(T))
    
    # Vega usually expressed as change per 1% change in volatility
    vega = (S * N_prime_d1 * math.sqrt(T)) / 100.0
    
    # Theta usually expressed as change per 1 day change in time
    theta = (- (S * N_prime_d1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N_d2) / 365.0

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4)
    }

def get_theoretical_atm_greeks(S: float, trailing_vol: float, days_to_expiration: int):
    """
    Convenience function: computes ATM greeks using current stock price S,
    a given trailing volatility, and a specific DTE.
    S = K (At The Money)
    """
    T = days_to_expiration / 365.0
    # K is equal to S for ATM
    return calculate_black_scholes_greeks(S=S, K=S, T=T, sigma=trailing_vol)
