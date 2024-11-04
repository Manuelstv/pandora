"""
Kent Distribution Implementation

This module provides an implementation of the Kent distribution on a sphere,
along with related utility functions for spherical coordinates and matrix operations.

The algorithms are partially based on methods described in:
[The Fisher-Bingham Distribution on the Sphere, John T. Kent
Journal of the Royal Statistical Society. Series B (Methodological)
Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
Article Stable URL: http://www.jstor.org/stable/2984712]
"""

import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.distributions import Normal, Uniform


def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Perform matrix multiplication with gradient tracking.

    Args:
        a: First input tensor
        b: Second input tensor

    Returns:
        Result of matrix multiplication
    """
    result = torch.matmul(a, b)
    if result.requires_grad:
        result.register_hook(lambda grad: None)
    return result


def norm(x: Tensor, axis: Optional[int] = None) -> Tensor:
    """
    Compute the L2 norm along a specified axis.

    Args:
        x: Input tensor
        axis: Axis along which to compute the norm (default: 0)

    Returns:
        L2 norm of the input tensor
    """
    if axis is None:
        axis = 0
    return torch.sqrt(torch.sum(x * x, dim=axis))


def generate_arbitrary_orthogonal_unit_vector(x: Tensor) -> Tensor:
    """
    Generate an arbitrary orthogonal unit vector to the input vector.

    Args:
        x: Input vector

    Returns:
        Orthogonal unit vector
    """
    v1 = torch.cross(x, torch.tensor([1.0, 0.0, 0.0]))
    v2 = torch.cross(x, torch.tensor([0.0, 1.0, 0.0]))
    v3 = torch.cross(x, torch.tensor([0.0, 0.0, 1.0]))
    v1n, v2n, v3n = norm(v1), norm(v2), norm(v3)
    v = [v1, v2, v3][torch.argmax(torch.tensor([v1n, v2n, v3n]))]
    return v / norm(v)


def kent(
    alpha: float,
    eta: float,
    psi: float,
    kappa: float,
    beta: float
) -> 'KentDistribution':
    """
    Generate a Kent distribution based on spherical coordinates.

    Args:
        alpha: Spherical coordinate alpha
        eta: Spherical coordinate eta
        psi: Spherical coordinate psi
        kappa: Concentration parameter
        beta: Ovalness parameter

    Returns:
        KentDistribution object
    """
    gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(
        alpha, eta, psi
    )
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)


def kent2(
    gamma1: Tensor,
    gamma2: Tensor,
    gamma3: Tensor,
    kappa: float,
    beta: float
) -> 'KentDistribution':
    """
    Generate a Kent distribution using orthonormal vectors.

    Args:
        gamma1: First orthonormal vector
        gamma2: Second orthonormal vector
        gamma3: Third orthonormal vector
        kappa: Concentration parameter
        beta: Ovalness parameter

    Returns:
        KentDistribution object
    """
    return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)


def kent4(Gamma: Tensor, kappa: float, beta: float) -> 'KentDistribution':
    """
    Generate a Kent distribution using a matrix of orthonormal vectors.

    Args:
        Gamma: Matrix of orthonormal vectors
        kappa: Concentration parameter
        beta: Ovalness parameter

    Returns:
        KentDistribution object
    """
    gamma1, gamma2, gamma3 = Gamma[:, 0], Gamma[:, 1], Gamma[:, 2]
    return kent2(gamma1, gamma2, gamma3, kappa, beta)


class KentDistribution:
    """
    Kent Distribution class for spherical statistics.
    """

    minimum_value_for_kappa = 1E-6

    @staticmethod
    def create_matrix_H(alpha: Tensor, eta: Tensor) -> Tensor:
        """
        Create matrix H from spherical coordinates.

        Args:
            alpha: Spherical coordinate alpha
            eta: Spherical coordinate eta

        Returns:
            Matrix H
        """
        device = alpha.device
        dtype = alpha.dtype
        return torch.stack([
            torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.tensor(0.0, dtype=dtype, device=device)]),
            torch.stack([torch.sin(alpha) * torch.cos(eta), torch.cos(alpha) * torch.cos(eta), -torch.sin(eta)]),
            torch.stack([torch.sin(alpha) * torch.sin(eta), torch.cos(alpha) * torch.sin(eta), torch.cos(eta)])
        ])

    @staticmethod
    def create_matrix_Ht(alpha: Tensor, eta: Tensor) -> Tensor:
        """
        Create transpose of matrix H from spherical coordinates.

        Args:
            alpha: Spherical coordinate alpha
            eta: Spherical coordinate eta

        Returns:
            Transpose of matrix H
        """
        return torch.transpose(KentDistribution.create_matrix_H(alpha, eta), 0, 1)

    @staticmethod
    def create_matrix_K(psi: Tensor) -> Tensor:
        """
        Create matrix K from spherical coordinate psi.

        Args:
            psi: Spherical coordinate psi

        Returns:
            Matrix K
        """
        return torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(psi), -torch.sin(psi)],
            [0.0, torch.sin(psi), torch.cos(psi)]
        ])

    @staticmethod
    def create_matrix_Kt(psi: Tensor) -> Tensor:
        """
        Create transpose of matrix K from spherical coordinate psi.

        Args:
            psi: Spherical coordinate psi

        Returns:
            Transpose of matrix K
        """
        return torch.transpose(KentDistribution.create_matrix_K(psi), 0, 1)

    @staticmethod
    def create_matrix_Gamma(alpha: Tensor, eta: Tensor, psi: Tensor) -> Tensor:
        """
        Create matrix Gamma from spherical coordinates.

        Args:
            alpha: Spherical coordinate alpha
            eta: Spherical coordinate eta
            psi: Spherical coordinate psi

        Returns:
            Matrix Gamma
        """
        H = KentDistribution.create_matrix_H(alpha, eta)
        K = KentDistribution.create_matrix_K(psi)
        return matrix_multiply(H, K)

    @staticmethod
    def create_matrix_Gammat(alpha: Tensor, eta: Tensor, psi: Tensor) -> Tensor:
        """
        Create transpose of matrix Gamma from spherical coordinates.

        Args:
            alpha: Spherical coordinate alpha
            eta: Spherical coordinate eta
            psi: Spherical coordinate psi

        Returns:
            Transpose of matrix Gamma
        """
        return torch.transpose(KentDistribution.create_matrix_Gamma(alpha, eta, psi), 0, 1)

    @staticmethod
    def spherical_coordinates_to_gammas(
        alpha: Tensor,
        eta: Tensor,
        psi: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert spherical coordinates to gamma vectors.

        Args:
            alpha: Spherical coordinate alpha
            eta: Spherical coordinate eta
            psi: Spherical coordinate psi

        Returns:
            Tuple of gamma1, gamma2, gamma3 vectors
        """
        Gamma = KentDistribution.create_matrix_Gamma(alpha, eta, psi)
        return Gamma[:, 0], Gamma[:, 1], Gamma[:, 2]

    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert gamma1 vector to spherical coordinates alpha and eta.

        Args:
            gamma1: First gamma vector

        Returns:
            Tuple of alpha and eta
        """
        def safe_arccos(x: Tensor, eps: float = 1e-6) -> Tensor:
            return torch.arccos(torch.clamp(x, -1 + eps, 1 - eps))

        def safe_arctan2(y: Tensor, x: Tensor, eps: float = 1e-6) -> Tensor:
            return torch.atan2(y, x + eps * (x == 0).float())

        alpha = safe_arccos(gamma1[0])
        eta = safe_arctan2(gamma1[2], gamma1[1])
        return alpha, eta

    @staticmethod
    def gammas_to_spherical_coordinates(
        gamma1: Tensor,
        gamma2: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert gamma vectors to spherical coordinates.

        Args:
            gamma1: First gamma vector
            gamma2: Second gamma vector

        Returns:
            Tuple of alpha, eta, and psi
        """
        alpha, eta = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
        Ht = KentDistribution.create_matrix_Ht(alpha, eta)
        u = matrix_multiply(Ht, gamma2.reshape(3, 1))
        psi = torch.atan2(u[2][0], u[1][0])
        return alpha, eta, psi

    def __init__(
        self,
        gamma1: Tensor,
        gamma2: Tensor,
        gamma3: Tensor,
        kappa: float,
        beta: float
    ):
        """
        Initialize the Kent Distribution.

        Args:
            gamma1: First orthonormal vector
            gamma2: Second orthonormal vector
            gamma3: Third orthonormal vector
            kappa: Concentration parameter
            beta: Ovalness parameter
        """
        self.gamma1 = torch.tensor(gamma1, dtype=torch.float64)
        self.gamma2 = torch.tensor(gamma2, dtype=torch.float64)
        self.gamma3 = torch.tensor(gamma3, dtype=torch.float64)
        self.kappa = float(kappa)
        self.beta = float(beta)

        self.alpha, self.eta, self.psi = KentDistribution.gammas_to_spherical_coordinates(
            self.gamma1, self.gamma2
        )

        for gamma in (gamma1, gamma2, gamma3):
            assert len(gamma) == 3, "Gamma vectors must have length 3"

        self._cached_rvs = torch.empty((0, 3), dtype=torch.float64)

    @property
    def Gamma(self) -> Tensor:
        """
        Get the Gamma matrix.

        Returns:
            Gamma matrix
        """
        return self.create_matrix_Gamma(self.alpha, self.eta, self.psi)
  
    def normalize(
        self,
        cache: Dict[Tuple[float, float], float] = {},
        return_num_iterations: bool = False,
        approximate: bool = True
    ) -> Union[float, Tuple[float, int]]:
        """
        Calculate the normalization constant of the Kent distribution.

        Args:
            cache: Cache for storing computed values
            return_num_iterations: Whether to return the number of iterations
            approximate: Whether to use approximation for small beta/kappa ratios

        Returns:
            Normalization constant (and number of iterations if requested)
        """
        k, b = self.kappa, self.beta
        if (k, b) not in cache:
            if approximate and (2 * b) / k < 1:
                result = torch.exp(k) * ((k - 2 * b) * (k + 2 * b)) ** (-0.5)
            else:
                G = torch.special.gamma
                I = torch.special.i0
                result = 0.0
                j = 0
                if torch.isclose(torch.tensor(b), torch.tensor(0.0)):
                    result = ((0.5 * k) ** (-2 * j - 0.5)) * I(2 * j + 0.5, k)
                    result /= G(j + 1)
                    result *= G(j + 0.5)
                else:
                    while True:
                        a = torch.exp(torch.log(b) * 2 * j + torch.log(0.5 * k) * (-2 * j - 0.5)) * I(2 * j + 0.5, k)
                        a /= G(j + 1)
                        a *= G(j + 0.5)
                        result += a
                        j += 1
                        if abs(a) < abs(result) * 1E-12 and j > 5:
                            break
            cache[k, b] = 2 * torch.pi * result
        
        if return_num_iterations:
            return cache[k, b], j
        else:
            return cache[k, b]

    def log_normalize(self, return_num_iterations: bool = False) -> Union[float, Tuple[float, int]]:
        """
        Calculate the logarithm of the normalization constant.

        Args:
            return_num_iterations: Whether to return the number of iterations

        Returns:
            Logarithm of the normalization constant (and number of iterations if requested)
        """
        if return_num_iterations:
            normalize, num_iter = self.normalize(return_num_iterations=True)
            return torch.log(normalize), num_iter
        else:
            return torch.log(self.normalize())

    def pdf_max(self, normalize: bool = True) -> float:
        """
        Calculate the maximum value of the probability density function.

        Args:
            normalize: Whether to normalize the result

        Returns:
            Maximum value of the PDF
        """
        return torch.exp(self.log_pdf_max(normalize))

    def log_pdf_max(self, normalize: bool = True) -> float:
        """
        Calculate the maximum value of the log probability density function.

        Args:
            normalize: Whether to normalize the result

        Returns:
            Maximum value of the log PDF
        """
        if self.beta == 0.0:
            x = 1
        else:
            x = self.kappa * 1.0 / (2 * self.beta)
        if x > 1.0:
            x = 1
        fmax = self.kappa * x + self.beta * (1 - x**2)
        if normalize:
            return fmax - self.log_normalize()
        else:
            return fmax

    def pdf(self, xs: Tensor, normalize: bool = True) -> Tensor:
        """
        Calculate the probability density function for given points.

        Args:
            xs: Input points (N x 3 or N x M x 3 or N x M x P x 3 etc.)
            normalize: Whether to normalize the result

        Returns:
            PDF values for the input points
        """
        return torch.exp(self.log_pdf(xs, normalize))

    def log_pdf(self, xs: Tensor, normalize: bool = True) -> Tensor:
        """
        Calculate the log probability density function for given points.

        Args:
            xs: Input points (N x 3 or N x M x 3 or N x M x P x 3 etc.)
            normalize: Whether to normalize the result

        Returns:
            Log PDF values for the input points
        """
        axis = len(xs.shape) - 1
        g1x = torch.sum(self.gamma1 * xs, axis)
        g2x = torch.sum(self.gamma2 * xs, axis)
        g3x = torch.sum(self.gamma3 * xs, axis)
        k, b = self.kappa, self.beta

        f = k * g1x + b * (g2x**2 - g3x**2)
        if normalize:
            return f - self.log_normalize()
        else:
            return f

    def pdf_prime(self, xs: Tensor, normalize: bool = True) -> Tensor:
        """
        Calculate the derivative of the PDF with respect to kappa and beta.

        Args:
            xs: Input points
            normalize: Whether to normalize the result

        Returns:
            Derivative of the PDF
        """
        return self.pdf(xs, normalize) * self.log_pdf_prime(xs, normalize)

    def log_pdf_prime(self, xs: Tensor, normalize: bool = True) -> Tensor:
        """
        Calculate the derivative of the log PDF with respect to kappa and beta.

        Args:
            xs: Input points
            normalize: Whether to normalize the result

        Returns:
            Derivative of the log PDF
        """
        axis = len(xs.shape) - 1
        g1x = torch.sum(self.gamma1 * xs, axis)
        g2x = torch.sum(self.gamma2 * xs, axis)
        g3x = torch.sum(self.gamma3 * xs, axis)
        k, b = self.kappa, self.beta

        dfdk = g1x
        dfdb = g2x**2 - g3x**2
        df = torch.stack([dfdk, dfdb])
        if normalize:
            return torch.transpose(torch.transpose(df, 0, 1) - self.log_normalize_prime(), 0, 1)
        else:
            return df

    def normalize_prime(
        self,
        cache: Dict[Tuple[float, float], Tensor] = {},
        return_num_iterations: bool = False
    ) -> Union[Tensor, Tuple[Tensor, int]]:
        """
        Calculate the derivative of the normalization factor with respect to kappa and beta.

        Args:
            cache: Cache for storing computed values
            return_num_iterations: Whether to return the number of iterations

        Returns:
            Derivative of the normalization factor (and number of iterations if requested)
        """
        k, b = self.kappa, self.beta
        if (k, b) not in cache:
            G = torch.special.gamma
            I = torch.special.i0
            dIdk = lambda v, z: torch.special.i0e(v, z)
            dcdk, dcdb = 0.0, 0.0
            j = 0
            if b == 0:
                dcdk = (
                    (G(j + 0.5) / G(j + 1)) *
                    ((-0.5 * j - 0.125) * (k) ** (-2 * j - 1.5)) *
                    (I(2 * j + 0.5, k))
                )
                dcdk += (
                    (G(j + 0.5) / G(j + 1)) *
                    ((0.5 * k) ** (-2 * j - 0.5)) *
                    (dIdk(2 * j + 0.5, k))
                )
                dcdb = 0.0
            else:
                while True:
                    dk = ((-1 * j - 0.25) * torch.exp(
                        torch.log(b) * 2 * j +
                        torch.log(0.5 * k) * (-2 * j - 1.5)
                    ) * I(2 * j + 0.5, k))
                    
                    dk += (torch.exp(
                        torch.log(b) * 2 * j +
                        torch.log(0.5 * k) * (-2 * j - 0.5)
                    ) * dIdk(2 * j + 0.5, k))
                    
                    dk /= G(j + 1)
                    dk *= G(j + 0.5)

                    db = (2 * j * torch.exp(
                        torch.log(b) * (2 * j - 1) +
                        torch.log(0.5 * k) * (-2 * j - 0.5)
                    ) * I(2 * j + 0.5, k))
                    
                    db /= G(j + 1)
                    db *= G(j + 0.5)
                    dcdk += dk
                    dcdb += db
                
                    j += 1
                    if abs(dk) < abs(dcdk) * 1E-12 and abs(db) < abs(dcdb) * 1E-12 and j > 5:
                        break
            
            cache[k, b] = 2 * torch.pi * torch.stack([dcdk, dcdb])
        
        if return_num_iterations:
            return cache[k, b], j
        else:
            return cache[k, b]

    def log_normalize_prime(self, return_num_iterations: bool = False) -> Union[Tensor, Tuple[Tensor, int]]:
        """
        Calculate the derivative of the logarithm of the normalization factor.

        Args:
            return_num_iterations: Whether to return the number of iterations

        Returns:
            Derivative of the log normalization factor (and number of iterations if requested)
        """
        if return_num_iterations:
            normalize_prime, num_iter = self.normalize_prime(return_num_iterations=True)
            return normalize_prime / self.normalize(), num_iter
        else:
            return self.normalize_prime() / self.normalize()

    def log_likelihood(self, xs: Tensor) -> Tensor:
        """
        Calculate the log likelihood for given points.

        Args:
            xs: Input points

        Returns:
            Log likelihood values
        """
        retval = self.log_pdf(xs)
        return torch.sum(retval, len(retval.shape) - 1)

    def log_likelihood_prime(self, xs: Tensor) -> Tensor:
        """
        Calculate the derivative of the log likelihood with respect to kappa and beta.

        Args:
            xs: Input points

        Returns:
            Derivative of the log likelihood
        """
        retval = self.log_pdf_prime(xs)
        if len(retval.shape) == 1:
            return retval
        else:
            return torch.sum(retval, len(retval.shape) - 1)

    def _rvs_helper(self) -> Tensor:
        """
        Helper function to generate random samples using rejection sampling.

        Returns:
            Random samples from the Kent distribution
        """
        num_samples = 10000
        xs = Normal(0, 1).sample((num_samples, 3))
        xs = xs / norm(xs, 1).reshape(num_samples, 1)
        pvalues = self.pdf(xs, normalize=False)
        fmax = self.pdf_max(normalize=False)
        return xs[Uniform(0, fmax).sample((num_samples,)) < pvalues]

    def rvs(self, n_samples: Optional[int] = None) -> Tensor:
        """
        Generate random samples from the Kent distribution.

        Args:
            n_samples: Number of samples to generate (None for a single sample)

        Returns:
            Random samples from the Kent distribution
        """
        num_samples = 1 if n_samples is None else n_samples
        rvs = self._cached_rvs
        while len(rvs) < num_samples:
            new_rvs = self._rvs_helper()
            rvs = torch.cat([rvs, new_rvs])
        if n_samples is None:
            self._cached_rvs = rvs[1:]
            return rvs[0]
        else:
            self._cached_rvs = rvs[num_samples:]
            retval = rvs[:num_samples]
            return retval

    def __repr__(self) -> str:
        """
        String representation of the Kent Distribution.

        Returns:
            String representation
        """
        return f"kent({self.alpha}, {self.eta}, {self.psi}, {self.kappa}, {self.beta})"


def project_equirectangular_to_sphere(u: Tensor, w: int, h: int) -> Tensor:
    """
    Project equirectangular coordinates to spherical coordinates.

    Args:
        u: Input coordinates
        w: Width of the equirectangular projection
        h: Height of the equirectangular projection

    Returns:
        Spherical coordinates
    """
    alpha = u[:, 1] * (torch.pi / float(h))
    eta = u[:, 0] * (2. * torch.pi / float(w))
    return torch.vstack([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ]).T


def project_sphere_to_equirectangular(x: Tensor, w: int, h: int) -> Tensor:
    """
    Project spherical coordinates to equirectangular coordinates.

    Args:
        x: Input spherical coordinates
        w: Width of the equirectangular projection
        h: Height of the equirectangular projection

    Returns:
        Equirectangular coordinates
    """
    alpha = torch.squeeze(torch.arccos(torch.clamp(x[:, 0], -1, 1)))
    eta = torch.squeeze(torch.atan2(x[:, 2], x[:, 1]))
    eta[eta < 0] += 2 * torch.pi
    return torch.vstack([eta * float(w) / (2 * torch.pi), alpha * float(h) / torch.pi])

def get_me_matrix_torch(xs: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate the dispersion matrix and mean direction for given points.

    Args:
        xs: Input points

    Returns:
        Tuple of dispersion matrix and mean direction
    """
    lenxs = len(xs)
    xbar = torch.mean(xs, 0)
    S = torch.mean(xs.reshape((lenxs, 3, 1)) * xs.reshape((lenxs, 1, 3)), 0)
    return S, xbar

def kent_me_matrix_torch(S_torch: Tensor, xbar_torch: Tensor) -> Tensor:
    """
    Calculate Kent distribution parameters from dispersion matrix and mean direction.

    Args:
        S_torch: Dispersion matrix
        xbar_torch: Mean direction

    Returns:
        Kent distribution parameters (psi, alpha, eta, kappa, beta)
    """
    S_torch = S_torch.float().requires_grad_(True)
    xbar_torch = xbar_torch.float().requires_grad_(True)

    gamma1 = xbar_torch / norm(xbar_torch, axis=0)
    alpha, eta = KentDistribution.gamma1_to_spherical_coordinates(gamma1)

    H = KentDistribution.create_matrix_H(alpha, eta)
    Ht = KentDistribution.create_matrix_Ht(alpha, eta)

    B = matrix_multiply(Ht, matrix_multiply(S_torch, H))

    alpha_hat = 0.5 * torch.atan2(2 * B[1, 2], B[1, 1] - B[2, 2])

    K = torch.stack([
        torch.tensor([1, 0, 0], dtype=S_torch.dtype, device=S_torch.device),
        torch.stack([torch.tensor(0, dtype=S_torch.dtype, device=S_torch.device), torch.cos(alpha_hat), -torch.sin(alpha_hat)]),
        torch.stack([torch.tensor(0, dtype=S_torch.dtype, device=S_torch.device), torch.sin(alpha_hat), torch.cos(alpha_hat)])
    ])

    G = matrix_multiply(H, K)
    Gt = torch.transpose(G, 0, 1)
    T = matrix_multiply(Gt, matrix_multiply(S_torch, G))

    r1 = norm(xbar_torch)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33

    min_kappa = torch.tensor(1E-6, dtype=S_torch.dtype, device=S_torch.device)
    kappa = torch.max(min_kappa, 1.0 / (2.0 - 2.0 * r1 - r2) + 1.0 / (2.0 - 2.0 * r1 + r2))
    beta = 0.5 * (1.0 / (2.0 - 2.0 * r1 - r2) - 1.0 / (2.0 - 2.0 * r1 + r2))

    gamma1 = G[:, 0]
    gamma2 = G[:, 1]
    gamma3 = G[:, 2]

    psi, alpha, eta = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)

    #print(kappa, beta)
    return torch.stack([psi, alpha, eta, kappa, beta])


def gradient_check():
    """
    Perform gradient checks for the kent_me_matrix_torch function.
    """
    S_torch = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    xbar_torch = torch.randn(3, dtype=torch.float64, requires_grad=True)

    result = kent_me_matrix_torch(S_torch, xbar_torch)
    loss = result.sum()
    loss.backward(retain_graph=True)

    assert S_torch.grad is not None, "Gradient for S_torch is None"
    assert xbar_torch.grad is not None, "Gradient for xbar_torch is None"
    assert torch.all(S_torch.grad != 0), "Gradient for S_torch is zero"
    assert torch.all(xbar_torch.grad != 0), "Gradient for xbar_torch is zero"

    S_torch.grad.zero_()
    xbar_torch.grad.zero_()
    loss.backward(retain_graph=True)
    assert torch.allclose(S_torch.grad, S_torch.grad), "Inconsistent gradients for S_torch"
    assert torch.allclose(xbar_torch.grad, xbar_torch.grad), "Inconsistent gradients for xbar_torch"

    epsilon = 1e-5
    S_torch_fd = S_torch.clone().detach().requires_grad_(True)
    xbar_torch_fd = xbar_torch.clone().detach().requires_grad_(True)
    result_fd = kent_me_matrix_torch(S_torch_fd, xbar_torch_fd)
    loss_fd = result_fd.sum()
    loss_fd.backward(retain_graph=True)

    numerical_grad_S = torch.zeros_like(S_torch)
    numerical_grad_xbar = torch.zeros_like(xbar_torch)

    for i in range(S_torch.numel()):
        S_torch_flat = S_torch.view(-1).clone().detach().requires_grad_(True)
        S_torch_perturbed_pos = S_torch_flat.clone()
        S_torch_perturbed_pos[i] += epsilon
        S_torch_perturbed_pos = S_torch_perturbed_pos.view(S_torch.size())
        result_pos = kent_me_matrix_torch(S_torch_perturbed_pos, xbar_torch).sum()
        
        S_torch_perturbed_neg = S_torch_flat.clone()
        S_torch_perturbed_neg[i] -= epsilon
        S_torch_perturbed_neg = S_torch_perturbed_neg.view(S_torch.size())
        result_neg = kent_me_matrix_torch(S_torch_perturbed_neg, xbar_torch).sum()
        
        numerical_grad = (result_pos - result_neg) / (2 * epsilon)
        numerical_grad_S.view(-1)[i] = numerical_grad

    for i in range(xbar_torch.numel()):
        xbar_torch_flat = xbar_torch.view(-1).clone().detach().requires_grad_(True)
        xbar_torch_perturbed_pos = xbar_torch_flat.clone()
        xbar_torch_perturbed_pos[i] += epsilon
        xbar_torch_perturbed_pos = xbar_torch_perturbed_pos.view(xbar_torch.size())
        result_pos = kent_me_matrix_torch(S_torch, xbar_torch_perturbed_pos).sum()
        
        xbar_torch_perturbed_neg = xbar_torch_flat.clone()
        xbar_torch_perturbed_neg[i] -= epsilon
        xbar_torch_perturbed_neg = xbar_torch_perturbed_neg.view(xbar_torch.size())
        result_neg = kent_me_matrix_torch(S_torch, xbar_torch_perturbed_neg).sum()
        
        numerical_grad = (result_pos - result_neg) / (2 * epsilon)
        numerical_grad_xbar.view(-1)[i] = numerical_grad

    assert torch.allclose(S_torch.grad, numerical_grad_S, atol=1e-4), "Gradient check failed for S_torch"
    assert torch.allclose(xbar_torch.grad, numerical_grad_xbar, atol=1e-4), "Gradient check failed for xbar_torch"

    print("Numerical gradients match analytical gradients.")


class SimpleModel(nn.Module):
    """
    A simple model with trainable parameters for Kent distribution.
    """
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.bias = nn.Parameter(torch.randn(3))

    def forward(self, S: Tensor, xbar: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            S: Dispersion matrix
            xbar: Mean direction

        Returns:
            Kent distribution parameters
        """
        S_transformed = torch.matmul(S, self.weight) + self.bias
        xbar_transformed = torch.matmul(xbar, self.weight) + self.bias
        return kent_me_matrix_torch(S_transformed, xbar_transformed)


def minimal_example():
    """
    Run a minimal example of training a SimpleModel with Kent distribution.
    """
    batch_size = 1
    S_torch = torch.randn(3, 3, requires_grad=True)
    xbar_torch = torch.randn(3, requires_grad=True)

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(S_torch, xbar_torch)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    gradient_check()
    minimal_example()