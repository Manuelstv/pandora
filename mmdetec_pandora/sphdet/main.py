import torch
from iou.sph_iou_calculator import SphOverlaps2D

if __name__ == "__main__":
    # Define the two tensors
    bbox1 = torch.tensor([40, 50, 35, 55])
    bbox2 = torch.tensor([35, 20, 37, 50])

    # Initialize the SphOverlaps2D object with the 'fov_iou' backend
    iou_calculator = SphOverlaps2D(backend='fov_iou')

    # Compute the FOVIoU
    fov_iou_value = iou_calculator(bbox1.unsqueeze(0), bbox2.unsqueeze(0), mode='iou')

    print("FOVIoU:", fov_iou_value.item())