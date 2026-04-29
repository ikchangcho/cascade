from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

def p(alpha, beta, N):
    """
    Compute p(alpha, beta, N) = 1/sqrt(2πNα(1-α)) * (β/α)^(αN) * ((1-β)/(1-α))^(N-αN)
    """
    numerator = (beta / alpha) ** (alpha * N) * ((1 - beta) / (1 - alpha)) ** (N - alpha * N)
    denominator = np.sqrt(2 * np.pi * N * alpha * (1 - alpha))
    return numerator / denominator

def p_gaussian(alpha, beta, N):
    """
    Gaussian approximation: p(α, β) = (N / sqrt(2π(β - β²))) * exp(-(α - β)² N² / (2(β - β²)))
    """
    variance = beta * (1 - beta)
    normalization = N / np.sqrt(2 * np.pi * variance)
    exponent = -((alpha - beta) ** 2) * (N ** 2) / (2 * variance)
    return normalization * np.exp(exponent)

# Parameters
for N in [100, 10]:
    for beta in [0.1, 0.5, 0.9]:
        # Generate alpha values (avoid 0 and 1 to prevent division issues)
        alpha_values = np.linspace(0.01, 0.99, 1000)
        p_values = p(alpha_values, beta, N)
        p_gaussian_values = p_gaussian(alpha_values, beta, N)

        # Exact discrete binomial distribution
        k = np.arange(0, N + 1)
        binomial_probs = binom.pmf(k, N, beta)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, p_values, label=f'p(α, β, N)', linewidth=2)
        plt.plot(alpha_values, p_gaussian_values, label='Gaussian approximation', linewidth=2, linestyle='--', color='red')
        plt.bar(k / N, binomial_probs, width=0.005, alpha=0.6, label='Exact Discrete Distribution')
        plt.xlabel('α')
        plt.ylabel(r'$p(\alpha)$')
        plt.title(f'Probability density for β={beta}, N={N}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()