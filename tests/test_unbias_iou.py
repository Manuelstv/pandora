import utils.ext_import

import torch
from sphdet.iou.unbiased_iou_bfov import Sph as NpSph
from sphdet.iou.unbias_iou_torch import Sph as TorchSph

from utils.generate_data import generate_boxes
from utils.timer import Timer


def test_unbias_iou_np_torch():
    N = 100
    is_aligned=True
    eps = 1e-5
    device = 'cpu'

    with Timer('LoadData', sync=True):
        bboxes1 = generate_boxes(N, version='rad').to(device)
        bboxes2 = generate_boxes(N, version='rad').to(device)

    with Timer('TorchSph', sync=True):
        overlaps1 = TorchSph().sphIoU(bboxes1, bboxes2, is_aligned)

    with Timer('NumpySph', sync=True):
        overlaps2 = NpSph().sphIoU(bboxes1, bboxes2, is_aligned)

    err = torch.abs(overlaps1.to(device) - torch.as_tensor(overlaps2).to(device))
    print('err: mean={:.6f}, var={:.6f}, median={:.6f}, max={:.6f}, min={:.6f}' \
        .format(err.mean().item(), err.var().item(), err.median().item(), \
                err.max().item(), err.min().item()))

    torch.set_printoptions(precision=8, sci_mode=False)
    id = err.argmax().item()
    print('id={}, box1={}, box2={}, iou1={}, iou2={}'.format(id, bboxes1[id], bboxes2[id], overlaps1[id], overlaps2[id]))
    assert err.mean().item() < eps


if __name__ == '__main__':
    test_unbias_iou_np_torch()
