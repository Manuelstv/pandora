import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
from sphdet.losses.kent_kld import get_kld, jiter_spherical_bboxes
import numpy as np
import math


def bfov_to_kent(annotations, epsilon=1e-6):
    if annotations.ndim == 1:
        annotations = annotations.unsqueeze(0)

    eta = np.pi*annotations[:, 0] / 180.0
    alpha = np.pi * annotations[:, 1] / 180.0

    fov_theta = annotations[:, 2]
    fov_phi = annotations[:, 3]

    width = torch.sin(alpha) * torch.deg2rad(fov_theta)
    height = torch.deg2rad(fov_phi)

    varphi = (height**2) / 12 + epsilon
    vartheta = (width**2) / 12 + epsilon

    kappa = 0.5 * (1 / varphi + 1 / vartheta)
    beta = torch.abs(0.25 * (1 / vartheta - 1 / varphi))

    beta = torch.where((2*beta)/kappa>0.97,
                       beta*0.97,
                       beta)

    psi = 0*torch.where(
        annotations[:, 4]<0,
        np.pi + annotations[:, 4],
        annotations[:, 4]
    )


    '''psi = torch.where(
        vartheta > varphi,
        np.pi / 2 + psi,
        psi
    )'''

    kent_dist = torch.stack([eta, alpha, kappa, beta, psi, fov_theta, fov_phi], dim=1)

    return kent_dist

def haversine_distance(theta1_deg, phi1_deg, theta2_deg, phi2_deg):
    """
    Compute distances between corresponding points only (not pairwise matrix).
    All inputs should have the same shape [N].
    """
    theta1 = theta1_deg - torch.pi/2
    phi1 = phi1_deg - torch.pi
    theta2 = theta2_deg - torch.pi/2
    phi2 = phi2_deg - torch.pi

    delta_theta = theta1 - theta2
    delta_phi = phi1 - phi2

    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_theta/2)**2
    return 2 * torch.atan(torch.sqrt(a/(1-a)))

class SphBox2KentTransform:
    def __init__(self):
        self.transform = _sph_box2kent_transform
    def __call__(self, boxes):
        return self.transform(boxes)

def _sph_box2kent_transform(boxes):
    return bfov_to_kent(boxes)

@LOSSES.register_module()
class KentLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super().__init__()
        #assert mode in ['iou', 'giou', 'diou', 'ciou']
        #self.mode = mode
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.transform = SphBox2KentTransform()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)

        pred, target = jiter_spherical_bboxes(pred, target)

        kent_pred = self.transform(pred)
        kent_target = self.transform(target)

        loss = self.loss_weight * kent_loss(
            kent_pred,
            kent_target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss

@weighted_loss
def kent_loss(y_pred, y_true, eps = 1e-6):
    # Ensure inputs are 2D[]
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)

    pred = y_pred.double()
    true = y_true.double()

    kld_pt = get_kld(pred, true)
    kld_tp = get_kld(true, pred)

    kld_pt = torch.clamp(kld_pt, min =0)
    kld_tp = torch.clamp(kld_tp, min =0)

    jsd = (kld_pt+kld_tp)
    const = 1.
    jsd_iou = 1 / (const + jsd**2)
    #jsd_iou = torch.exp(-1*jsd)

    #eta, alpha, kappa, beta, psi, fov_theta, fov_

    w2, h2 = y_pred[:,5], y_pred[:,6]
    w1, h1 = y_true[:,5], y_true[:,6]

    factor = 4 / torch.pi ** 2

    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    #Should we use masking like sph2pob?
    with torch.no_grad():
        alpha = v / (1 - jsd_iou + v + eps)

    c_center = torch.clamp(haversine_distance(true[:, 0], true[:, 1], pred[:, 0], pred[:, 1]), min =0, max = torch.pi)
    rho_squared = c_center**2

    w_true = torch.pi*(true[:, 5])/180
    h_true = torch.pi*(true[:, 6])/180
    w_pred = torch.pi*(pred[:, 5])/180
    h_pred = torch.pi*(pred[:, 6])/180

    true_bottom = true[:, 1] - h_true/2
    true_top = true[:, 1] + h_true/2
    true_left = true[:, 0] - w_true/2
    true_right = true[:, 0] + w_true/2

    pred_bottom = pred[:, 1] - h_pred/2
    pred_top = pred[:, 1] + h_pred/2
    pred_left = pred[:, 0] - w_pred/2
    pred_right = pred[:, 0] + w_pred/2

    enclose_bottom = torch.minimum(pred_bottom, true_bottom)
    enclose_top = torch.maximum(pred_top, true_top)
    enclose_left = torch.minimum(pred_left, true_left)
    enclose_right = torch.maximum(pred_right, true_right)

    c_encl = haversine_distance(enclose_left, enclose_bottom, enclose_right, enclose_top)
    c_squared = c_encl**2

    epsilon = 1e-6
    distance_penalty = rho_squared / (c_squared + epsilon)

    kld_loss = 1 - jsd_iou + alpha*v + distance_penalty
    return kld_loss


if __name__ == "__main__":
    pred = torch.tensor([[  2.2335,   1.7491, 331.7626, 146.6469]], dtype=torch.float32, requires_grad=True)#.half()
    target = torch.tensor([[5.4978, 2.3562, 1.2747, 0.4459]], dtype=torch.float32, requires_grad=True)#.half()
    loss = get_kld(target, pred)
    #loss.backward(retain_graph=True)
    print(loss)