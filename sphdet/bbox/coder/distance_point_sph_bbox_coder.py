import torch
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.coder.base_bbox_coder import BaseBBoxCoder
from ..box_formator import _sph2pix_box_transform, _pix2sph_box_transform, xywh2xyxy, xyxy2xywh


@BBOX_CODERS.register_module()
class DistancePointSphBBoxCoder(BaseBBoxCoder):
    """Distance Point BBox coder.

    This coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, clip_border=True, box_version=4, img_shape=None):
        super(BaseBBoxCoder, self).__init__()
        self.clip_border = clip_border
        self.box_version = box_version
        self.img_shape = img_shape
        

    def encode(self, points, gt_bboxes, max_dis=None, eps=0.1, img_shape=(512, 1024)):
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 4), The format is "xyxy"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 4).
        """
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == self.box_version
        img_shape = self.img_shape if self.img_shape else img_shape
        return bbox2distance(points, gt_bboxes, max_dis, eps, img_shape)

    def decode(self, points, pred_bboxes, max_shape=None, img_shape=(512, 1024)):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4)
                or (N, 4)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == self.box_version
        if self.clip_border is False:
            max_shape = None
        img_shape = self.img_shape if self.img_shape else img_shape
        return distance2bbox(points, pred_bboxes, max_shape, img_shape)


def distance2bbox(points, distance, max_shape=None, img_shape=(512, 1024)):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            bboxes = _pix2sph_box_transform(xyxy2xywh(bboxes), img_shape)
            if distance.size(-1) == 5:
                bboxes = torch.concat([bboxes, distance[..., [-1,]]], dim=-1)
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            raise NotImplementedError
            #return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    bboxes = _pix2sph_box_transform(xyxy2xywh(bboxes), img_shape)
    if distance.size(-1) == 5:
        bboxes = torch.concat([bboxes, distance[..., [-1,]]], dim=-1)
    return bboxes


def bbox2distance(points, bbox, max_dis=None, eps=0.1, img_shape=(512, 1024)):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    if bbox.size(-1) == 5:
        _bbox = bbox[:, :4]
        _angle = bbox[:, -1]
    else:
        _bbox = bbox
        _angle = None
    bbox = xywh2xyxy(_sph2pix_box_transform(_bbox, img_shape))
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    if _angle is not None:
        bbox = torch.stack([left, top, right, bottom, _angle], -1)
    else:
        bbox = torch.stack([left, top, right, bottom], -1)
    return bbox