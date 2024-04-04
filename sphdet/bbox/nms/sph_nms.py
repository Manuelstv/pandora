import torch

from sphdet.iou import unbiased_iou, naive_iou, sph2pob_standard_iou, sph2pob_efficient_iou


class SphNMS:
    def __init__(self, iou_calculator='sph2pob_efficient'):
        if iou_calculator == 'sph2pob_efficient':
            self.iou_calculator = sph2pob_efficient_iou
        elif iou_calculator == 'unbiased_iou':
            self.iou_calculator = unbiased_iou
        elif iou_calculator == 'naive_iou':
            self.iou_calculator = naive_iou
        else:
            raise NotImplemented('Not supported iou_calculator.')

    def __call__(self, boxes, scores, idxs, nms_cfg, class_agnostic=False):
        return sph_batched_nms(boxes, scores, idxs, nms_cfg, self.iou_calculator, class_agnostic)


def sph_batched_nms(boxes, scores, idxs, nms_cfg, iou_calculator, class_agnostic=False):
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)

    boxes_for_nms = boxes

    nms_type = nms_cfg_.pop('type', 'nms')
    split_thr = nms_cfg_.pop('split_thr', 10000)
    iou_threshold = nms_cfg_.pop('iou_threshold', 0.5)

    max_num = min(nms_cfg_.pop('max_num', boxes_for_nms.shape[0]), boxes_for_nms.shape[0])
    total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    #! Some type of nms would reweight the score, such as SoftNMS
    scores_after_nms = scores.new_zeros(scores.size())
    for id in torch.unique(idxs):
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = sph_nms_op(boxes_for_nms[mask], scores[mask], iou_threshold, iou_calculator)
        total_mask[mask[keep]] = True
        scores_after_nms[mask[keep]] = scores[mask[keep]]
    keep = total_mask.nonzero(as_tuple=False).view(-1)
    
    scores, inds = scores_after_nms[keep].sort(descending=True)
    keep = keep[inds]
    boxes = boxes[keep]
    
    keep = keep[:max_num]
    boxes = boxes[:max_num]
    scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep

def sph_nms_op(boxes, scores, iou_threshold, iou_calculator):
    var_dim = boxes.size(1)
    assert var_dim in [4, 5]
    B = torch.argsort(scores, descending=True)
    keep = [] 
    while B.numel() > 0:
        keep.append(B[0])
        if B.numel() == 1: break
        iou = iou_calculator(boxes[B[0], :].reshape(-1, var_dim),
                      boxes[B[1:], :].reshape(-1, var_dim)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)
