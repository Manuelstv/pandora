import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES
from sphdet.bbox.deg2kent_single import deg2kent_single



def debug_negative_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor, kld: torch.Tensor):
    """
    Debug function to identify which Kent distribution parameters lead to negative KLD values.
    
    Args:
        kent_pred (torch.Tensor): Predicted Kent distribution parameters (eta, alpha, psi, kappa, beta)
        kent_target (torch.Tensor): Target Kent distribution parameters (eta, alpha, psi, kappa, beta)
        kld (torch.Tensor): The calculated KLD matrix
    """
    # Find indices where KLD is negative
    negative_indices = torch.where(kld < 0)
    
    if len(negative_indices[0]) == 0:
        print("No negative KLD values found.")
        return
    
    print(f"\nFound {len(negative_indices[0])} negative KLD values")
    print("\nDetailed analysis of negative KLD cases:")
    
    for i, j in zip(negative_indices[0], negative_indices[1]):
        kld_value = kld[i, j]
        
        # Get corresponding parameters
        pred_params = kent_pred[i]
        target_params = kent_target[j]
        
        print(f"\nKLD Value: {kld_value:.6f}")
        print("\nPredicted parameters:")
        print(f"  eta: {pred_params[0]:.6f}")
        print(f"  alpha: {pred_params[1]:.6f}")
        print(f"  psi: {pred_params[2]:.6f}")
        print(f"  kappa: {pred_params[3]:.6f}")
        print(f"  beta: {pred_params[4]:.6f}")
        
        print("\nTarget parameters:")
        print(f"  eta: {target_params[0]:.6f}")
        print(f"  alpha: {target_params[1]:.6f}")
        print(f"  psi: {target_params[2]:.6f}")
        print(f"  kappa: {target_params[3]:.6f}")
        print(f"  beta: {target_params[4]:.6f}")
        
        # Calculate parameter differences
        diff = pred_params - target_params
        print("\nParameter differences (pred - target):")
        print(f"  eta diff: {diff[0]:.6f}")
        print(f"  alpha diff: {diff[1]:.6f}")
        print(f"  psi diff: {diff[2]:.6f}")
        print(f"  kappa diff: {diff[3]:.6f}")
        print(f"  beta diff: {diff[4]:.6f}")
        print("-" * 50)


class SphBox2KentTransform:
    def __init__(self, img_size):
        self.transform = _sph_box2kent_transform
        self.img_size = img_size
    def __call__(self, boxes):
        return self.transform(boxes, self.img_size)
    
def _sph_box2kent_transform(boxes, img_size):
    img_h, img_w = img_size
    return deg2kent_single(boxes, img_h, img_w)

def check_nan_inf(tensor: torch.Tensor, name: str):
    """
    Check for NaN and Inf values in a tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.
        name (str): The name of the tensor for error reporting.
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def radians_to_Q(psi: torch.Tensor, alpha: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to a Q matrix.
    
    Args:
        psi (torch.Tensor): The psi angles.
        alpha (torch.Tensor): The alpha angles.
        eta (torch.Tensor): The eta angles.
    
    Returns:
        torch.Tensor: The resulting Q matrix.
    """
    N = alpha.size(0)
    psi = psi.view(N, 1)
    alpha = alpha.view(N, 1)
    eta = eta.view(N, 1)
    
    gamma_1 = torch.cat([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_2 = torch.cat([
        -torch.cos(psi) * torch.sin(alpha),
        torch.cos(psi) * torch.cos(alpha) * torch.cos(eta) - torch.sin(psi) * torch.sin(eta),
        torch.cos(psi) * torch.cos(alpha) * torch.sin(eta) + torch.sin(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma_3 = torch.cat([
        torch.sin(psi) * torch.sin(alpha),
        -torch.sin(psi) * torch.cos(alpha) * torch.cos(eta) - torch.cos(psi) * torch.sin(eta),
        -torch.sin(psi) * torch.cos(alpha) * torch.sin(eta) + torch.cos(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma = torch.cat((gamma_1, gamma_2, gamma_3), dim=2)
    check_nan_inf(gamma, "gamma")
    return gamma

def log_approximate_c(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Approximate the c value based on kappa and beta.
    
    Args:
        kappa (torch.Tensor): The kappa values.
        beta (torch.Tensor): The beta values.
    
    Returns:
        torch.Tensor: The approximated c value.
    """
    epsilon = 1e-6  # Small value to avoid division by zero
    
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    product = term1 * term2 + epsilon
    
    # Calculate in log space to avoid numerical instability
    log_result = torch.log(torch.tensor(2 * torch.pi, device=kappa.device)) + kappa - 0.5 * torch.log(product)
    
    check_nan_inf(log_result, "approximate_c")
    return log_result

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    """
    Calculate the expected value of x based on gamma and c values.
    
    Args:
        gamma_a1 (torch.Tensor): The first gamma values.
        c (torch.Tensor): The c values.
        c_k (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The expected value of x.
    """
    const = (torch.exp(c_k - c)).view(-1, 1)
    result = const * gamma_a1
    check_nan_inf(result, "expected_x")
    return result

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    """
    Calculate the expected value of xx^T based on kappa, beta, Q_matrix, c, and c_k values.
    
    Args:
        kappa (torch.Tensor): The kappa values.
        beta (torch.Tensor): The beta values.
        Q_matrix (torch.Tensor): The Q matrix.
        c (torch.Tensor): The c values.
        c_k (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The expected value of xx^T.
    """
    c_kk = log_del_2_kappa(kappa, beta)
    c_beta = log_del_beta(kappa, beta)
    epsilon = 1e-6  # Small value to avoid division by zero

    lambda_1 = torch.exp(c_k - c)
    lambda_2 = 0.5*(1 - torch.exp(c_kk - c) + torch.exp(c_beta - c))
    lambda_3= 0.5*(1 - torch.exp(c_kk - c) - torch.exp(c_beta - c))


    lambdas = torch.stack([lambda_1, lambda_2, lambda_3], dim=-1)  # Shape: [N, 3]
    lambda_matrix = torch.diag_embed(lambdas)  # Shape: [N, 3, 3]

    Q_matrix_T = Q_matrix.transpose(-1, -2)  # Transpose the last two dimensions: [N, 3, 3]
    result = torch.matmul(Q_matrix, torch.matmul(lambda_matrix, Q_matrix_T))  # Shape: [N, 3, 3]
    check_nan_inf(result, "expected_xxT")
    return result

def beta_gamma_exxt_gamma(beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    """
    Calculate the product of beta, gamma, ExxT, and gamma.
    
    Args:
        beta (torch.Tensor): The beta values.
        gamma (torch.Tensor): The gamma values.
        ExxT (torch.Tensor): The expected value of xx^T.
    
    Returns:
        torch.Tensor: The result of the calculation.
    """
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    result = beta * result  # Shape: (N,)
    check_nan_inf(result, "beta_gamma_exxt_gamma")
    return result

def calculate_log_term(c_b, c_a):
    """
    Calculate the log term of the KLD matrix.
    
    Args:
        c_b (torch.Tensor): The c_b values.
        c_a (torch.Tensor): The c_a values.
    
    Returns:
        torch.Tensor: The log term of the KLD matrix.
    """
    result = torch.log(c_b.view(-1, 1) / c_a.view(1, -1)).T
    check_nan_inf(result, "calculate_log_term")
    return result

def calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a):
    """
    Calculate the kappa term of the KLD matrix.
    
    Args:
        kappa_a (torch.Tensor): The kappa_a values.
        gamma_a1 (torch.Tensor): The gamma_a1 values.
        kappa_b (torch.Tensor): The kappa_b values.
        gamma_b1 (torch.Tensor): The gamma_b1 values.
        Ex_a (torch.Tensor): The expected value of x.
    
    Returns:
        torch.Tensor: The kappa term of the KLD matrix.
    """
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1
    kappa_a_gamma_a1_expanded = kappa_a_gamma_a1.unsqueeze(1)
    kappa_b_gamma_b1_expanded = kappa_b_gamma_b1.unsqueeze(0)
    diff_kappa_term = kappa_a_gamma_a1_expanded - kappa_b_gamma_b1_expanded
    Ex_a_expanded = Ex_a.unsqueeze(1).expand(-1, diff_kappa_term.size(1), -1)
    result = torch.sum(diff_kappa_term * Ex_a_expanded, dim=-1)
    check_nan_inf(result, "calculate_kappa_term")
    return result

def log_del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    
    numerator = torch.log(-2 * torch.pi*(4 * beta**2 + kappa - kappa**2)) + kappa
    denominator = 1.5*torch.log(kappa - 2 * beta) + 1.5*torch.log(kappa + 2 * beta) + epsilon
    result = numerator - denominator
    
    check_nan_inf(result, "del_kappa")
    return result

def log_del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    
    numerator = torch.log(2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2)) + kappa
    denominator = 2.5*torch.log((kappa - 2 * beta)) + 2.5*torch.log((kappa + 2 * beta)) + epsilon
    result = numerator - denominator

    check_nan_inf(result, "del_2_kappa")
    return result

def log_del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-6
    
    numerator = torch.log(8 * torch.pi*kappa) + torch.log(beta)
    denominator = 1.5* torch.log(kappa - 2 * beta) + 1.5*torch.log(kappa + 2 * beta) + epsilon
    result = numerator - denominator

    check_nan_inf(result, "del_beta")
    return result

def calculate_beta_term(beta_a: torch.Tensor, gamma_a2: torch.Tensor, beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:
    """
    Calculate the beta term of the KLD matrix.
    
    Args:
        beta_a (torch.Tensor): The beta_a values.
        gamma_a2 (torch.Tensor): The gamma_a2 values.
        beta_b (torch.Tensor): The beta_b values.
        gamma_b2 (torch.Tensor): The gamma_b2 values.
        ExxT_a (torch.Tensor): The expected value of xx^T.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The beta term of the KLD matrix for beta_a and beta_b.
    """
    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze(-1)
    beta_a_term_1_expanded = beta_a_term_1.expand(-1, beta_b.size(0))

    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2
    beta_b_gamma_b2_expanded = beta_b_gamma_b2.unsqueeze(0)
    ExxT_a_expanded = ExxT_a.unsqueeze(1)
    product = beta_b_gamma_b2_expanded.unsqueeze(2) * ExxT_a_expanded
    result = product.sum(dim=-1)
    gamma_b2_expanded = gamma_b2.unsqueeze(0)
    beta_b_term_1 = torch.sum(result * gamma_b2_expanded, dim=-1)

    check_nan_inf(beta_a_term_1_expanded, "calculate_beta_term: beta_a_term_1_expanded")
    check_nan_inf(beta_b_term_1, "calculate_beta_term: beta_b_term_1")
    return beta_a_term_1_expanded, beta_b_term_1

def kld_matrix(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD matrix.
    
    Args:
        kappa_a (torch.Tensor): The kappa_a values.
        beta_a (torch.Tensor): The beta_a values.
        gamma_a1 (torch.Tensor): The gamma_a1 values.
        gamma_a2 (torch.Tensor): The gamma_a2 values.
        gamma_a3 (torch.Tensor): The gamma_a3 values.
        kappa_b (torch.Tensor): The kappa_b values.
        beta_b (torch.Tensor): The beta_b values.
        gamma_b1 (torch.Tensor): The gamma_b1 values.
        gamma_b2 (torch.Tensor): The gamma_b2 values.
        gamma_b3 (torch.Tensor): The gamma_b3 values.
        Ex_a (torch.Tensor): The expected value of x.
        ExxT_a (torch.Tensor): The expected value of xx^T.
        c_a (torch.Tensor): The c_a values.
        c_b (torch.Tensor): The c_b values.
        c_ka (torch.Tensor): The kappa values.
    
    Returns:
        torch.Tensor: The KLD matrix.
    """
    log_term = c_b-c_a
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)
    
    kld = log_term + ex_a_term+ beta_a_term_1_expanded - beta_b_term_1 - beta_a_term_2_expanded + beta_b_term_2
    check_nan_inf(kld, "kld_matrix")
    return kld

def get_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KLD between predicted and target Kent distributions.
    
    Args:
        kent_pred (torch.Tensor): The predicted Kent distribution parameters.
        kent_target (torch.Tensor): The target Kent distribution parameters.
    
    Returns:
        torch.Tensor: The KLD matrix.
    """
    eta_a, alpha_a, psi_a, kappa_a, beta_a = kent_pred[:, 0], kent_pred[:, 1], kent_pred[:, 2], kent_pred[:, 3], kent_pred[:, 4]
    Q_matrix_a = radians_to_Q(psi_a, alpha_a, eta_a)

    eta_b, alpha_b, psi_b, kappa_b, beta_b = kent_target[:, 0], kent_target[:, 1], kent_target[:, 2], kent_target[:, 3], kent_target[:, 4]
    Q_matrix_b = radians_to_Q(psi_b, alpha_b, eta_b)

    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, :, 0], Q_matrix_a[:, :, 1], Q_matrix_a[:, :, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, :, 0], Q_matrix_b[:, :, 1], Q_matrix_b[:, :, 2]

    c_a = log_approximate_c(kappa_a, beta_a)
    c_b = log_approximate_c(kappa_b, beta_b)
    c_ka = log_del_kappa(kappa_a, beta_a)

    ExxT_a = expected_xxT(kappa_a, beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)


    kld = kld_matrix(kappa_a, beta_a, gamma_a1, gamma_a2, gamma_a3,
                            kappa_b, beta_b, gamma_b1, gamma_b2, gamma_b3,
                            Ex_a, ExxT_a, c_a, c_b, c_ka)

    check_nan_inf(kld, "get_kld")
    return kld

def kent_loss(kent_pred: torch.Tensor, kent_target: torch.Tensor, const: float = 1.0) -> torch.Tensor:

    kld = torch.clamp(get_kld(kent_pred, kent_target), min =0)
    #debug_negative_kld(kent_pred, kent_target, kld)

    check_nan_inf(kld, "kld")
    
    result = 1 - 1 / (const + torch.sqrt(kld+1e-6))
    check_nan_inf(result, "kent_loss")
    return result


@LOSSES.register_module()
class KentLoss(nn.Module):
    """
    A PyTorch module for calculating the Kent loss.
    """
    def __init__(self, img_size, loss_weight=30.0):
        super(KentLoss, self).__init__()
        self.loss_weight = loss_weight
        self.transform = SphBox2KentTransform(img_size)
    
    def forward(self, pred, target, weight=None,
            avg_factor=None,
            reduction_override=None,
            **kwargs):
        """
        Forward pass for the Kent loss calculation.
        
        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The target values.
            weight (torch.Tensor, optional): The weight for loss calculation.
            avg_factor (float, optional): Average factor for loss calculation.
            reduction_override (str, optional): Override for reduction method.
            **kwargs: Additional arguments.
        
        Returns:
            torch.Tensor: The calculated loss.
        """
        
        kent_target = self.transform(target) #.squeeze()
        kent_pred = torch.clamp(self.transform(pred), min=0)  # Clamp to 0
        loss = kent_loss(kent_pred, kent_target)
            
        return loss
    
if __name__ == "__main__":
    pred = torch.tensor([1.0, -0.5, -0.1, -1.0], dtype=torch.float32, requires_grad=True)#.half()
    target = torch.tensor([0.0, 0.0, 40.0, 40.0], dtype=torch.float32, requires_grad=True)#.half()
    loss = KentLoss((1960, 980))(pred, target)
    loss.backward(retain_graph=True)
    print(loss)