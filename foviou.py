import torch
from numpy import deg2rad


def find_foviou(Bg, Bd):
    """
    Calculate the Field of View (FoV) Intersection over Union (IoU) matrix for all combinations of bounding boxes between two sets.

    This function computes the FoV-IoU for each pair of bounding boxes from two input sets, returning a matrix where each element (i, j) represents the IoU between the i-th box of the first set (Bg) and the j-th box of the second set (Bd).

    Parameters:
    - Bg (torch.Tensor): A tensor of shape [N, 4], where N is the number of bounding boxes in the ground truth set.
                         Each row represents a bounding box in the format deg2rad([theta, phi, alpha, beta]).
    - Bd (torch.Tensor): A tensor of shape [M, 4], where M is the number of bounding boxes in the detected set.
                         Each row represents a bounding box in the same format as Bg.

    Returns:
    - torch.Tensor: A matrix of shape [N, M], where each element (i, j) is the FoV-IoU between the i-th bounding box of Bg and the j-th bounding box of Bd.

    The FoV-IoU is calculated by first determining the area of intersection and the area of union for the field of view of each bounding box pair, then dividing the area of intersection by the area of union.

    Note: This function uses vectorized operations and broadcasting in PyTorch to efficiently compute the IoU matrix for large sets of bounding boxes.
    """
    
    # Assume Bg and Bd are of shape [N, 4] and [M, 4], where N and M are the number of bounding boxes in each set
    # Each row is [theta, phi, alpha, beta]
    
    # Expand Bg and Bd for vectorized operations across all combinations
    theta_g, phi_g, alpha_g, beta_g = Bg[:, None, :].expand(-1, Bd.shape[0], -1).unbind(-1)
    theta_d, phi_d, alpha_d, beta_d = Bd[None, :, :].expand(Bg.shape[0], -1, -1).unbind(-1)
    
    # Calculate FoV Area of Bg and Bd
    A_Bg = alpha_g * beta_g
    A_Bd = alpha_d * beta_d

    # Calculate FoV distance between every combination of Bg and Bd
    delta_fov = (theta_d - theta_g) * torch.cos((phi_g + phi_d) / 2)

    # Construct an approximate FoV Intersection for every combination
    theta_I_min = torch.max(-alpha_g / 2, delta_fov - alpha_d / 2)
    theta_I_max = torch.min(alpha_g / 2, delta_fov + alpha_d / 2)
    phi_I_min = torch.max(phi_g - beta_g / 2, phi_d - beta_d / 2)
    phi_I_max = torch.min(phi_g + beta_g / 2, phi_d + beta_d / 2)

    # Calculate the Area of the FoV Intersection and Union
    A_I = (theta_I_max - theta_I_min).clamp(min=0) * (phi_I_max - phi_I_min).clamp(min=0)
    A_U = A_Bg + A_Bd - A_I

    # Calculate the FoV-IoU, ensuring no division by zero
    FoV_IoU = A_I / A_U.clamp(min=1e-6)  # Add a small value to avoid division by zero

    return FoV_IoU
# Example usage with PyTorch tensors
set_1_torch = torch.tensor(deg2rad([[30, 60, 60, 60], [45, 50, 55, 60]]))
set_2_torch = torch.tensor(deg2rad([[30, 75, 26, 45], [75, 75, 26, 45]]))

#print(find_foviou(set_1_torch, set_2_torch))