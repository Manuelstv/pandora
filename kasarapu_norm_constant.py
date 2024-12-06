import numpy as np
from scipy.special import gamma, iv  # for Gamma function and modified Bessel function

class KentNormalization:
    """
    Class to compute the log normalization constant and its derivatives for the Kent (FB5) distribution.
    """
    def __init__(self, kappa, beta, epsilon=1e-6):
        """
        Initialize with kappa and beta parameters.
        
        Parameters:
        -----------
        kappa : float
            Concentration parameter κ ≥ 0
        beta : float 
            Ovalness parameter β, where 0 ≤ β < κ/2
        epsilon : float
            Convergence threshold for series summation
        """
        # Input validation
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if beta < 0 or beta >= kappa/2:
            raise ValueError("beta must satisfy 0 ≤ beta < kappa/2")
            
        self.kappa = kappa
        self.beta = beta
        self.epsilon = epsilon
        self.e = 2 * beta / kappa  # eccentricity
        self.delta_1 = 2 * np.pi * np.sqrt(2/kappa)
        self.log_delta_1 = np.log(self.delta_1)
        
    def compute_log_series(self, m=0):
        """
        Compute the logarithm sum S_1^(m) as defined in equation (21) of the paper.
        
        Parameters:
        -----------
        m : int
            Order of derivative (0 for log(c), 1 for first derivative, 2 for second derivative)
        
        Returns:
        --------
        float
            The computed logarithm sum
        """
        def compute_log_fj(j):
            """Compute log(f_j) for a given j"""
            # Calculate log(Γ(j + 1/2)/Γ(j + 1))
            log_gamma_ratio = np.log(gamma(j + 0.5)) - np.log(gamma(j + 1))
            
            # Calculate p = 2j + 1/2
            p = 2 * j + 0.5
            
            # Calculate log(e^(2j))
            log_e_term = 2 * j * np.log(self.e)
            
            # Calculate log(I_{p+m}(κ))
            log_bessel = np.log(iv(p + m, self.kappa))
            
            return log_gamma_ratio + log_e_term + log_bessel

        # Compute first term (j=0) to use as reference
        log_f0 = compute_log_fj(0)
        
        # Initialize sum
        curr_sum = np.exp(0)  # t_0 = 1
        
        j = 1
        while True:
            # Compute log(f_j)
            log_fj = compute_log_fj(j)
            
            # Compute t_j = f_j/f_0 = exp(log(f_j) - log(f_0))
            t_j = np.exp(log_fj - log_f0)
            
            # Update sum
            curr_sum += t_j
            
            # Check convergence
            if t_j / curr_sum < self.epsilon:
                break
                
            j += 1
            if j > 1000:
                raise RuntimeError("Series did not converge within 1000 iterations")
        
        # Return final result: log(δ_1) + log(f_0) + log(sum(t_j))
        return self.log_delta_1 + log_f0 + np.log(curr_sum)

    def log_c(self):
        """Compute log(c(κ,β))"""
        return self.compute_log_series(m=0)
    
    def log_c_kappa(self):
        """Compute log(∂c/∂κ)"""
        return self.compute_log_series(m=1)
    
    def log_c_kappa_kappa(self):
        """
        Compute log(∂²c/∂κ²) using the formula from the paper:
        log(c_κκ) = log(c_κ) + log(exp(S_1^(2) - S(c_κ)) + 1/κ)
        """
        log_c_k = self.log_c_kappa()
        S_2 = self.compute_log_series(m=2)
        
        return log_c_k + np.log(np.exp(S_2 - log_c_k) + 1/self.kappa)

def approximate_c(kappa, beta):
    """
    Approximate the log of c value based on asymptotic formula from Kent (1982).
    """
    epsilon = 1e-6
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    product = term1 * term2 + epsilon
    return np.log(2 * np.pi) + kappa - 0.5 * np.log(product)


def del_kappa(kappa, beta):
    """
    Calculate the derivative of kappa with respect to beta.
    
    Args:
        kappa (torch.Tensor): The kappa values.
        beta (torch.Tensor): The beta values.
    
    Returns:
        torch.Tensor: The derivative of kappa.
    """
    epsilon = 1e-6  # Small value to avoid division by zero

    numerator = -2 * np.pi * (4 * beta**2 + kappa - kappa**2) * np.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2) + epsilon  # Add epsilon to avoid division by zero
    result = np.log(numerator / denominator)
    return result

# Example usage
if __name__ == "__main__":
    # Example parameters
    kappa = 2.0
    beta = 0.1  # eccentricity = 0.5
    
    try:
        # Create normalization computer
        kent_norm = KentNormalization(kappa, beta)
        
        # Compute log(c) and its derivatives
        log_c = kent_norm.log_c()
        log_c_k = kent_norm.log_c_kappa()


        log_c_k_approx = del_kappa(kappa, beta)

        log_c_kk = kent_norm.log_c_kappa_kappa()
        
        # Compute approximation
        log_c_approx = approximate_c(kappa, beta)
        
        print(f"Log normalization constant:")
        print(f"  Exact:        {log_c:.6f}")
        print(f"  Approximated: {log_c_approx:.6f}")
        print(f"\nDerivatives:")
        print(f"  log(∂c/∂κ):   {log_c_k:.6f}")

        print(f"approx  log(∂c/∂κ):   {log_c_k_approx:.6f}")

        print(f"  log(∂²c/∂κ²): {log_c_kk:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")