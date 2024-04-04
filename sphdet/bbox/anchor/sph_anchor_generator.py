import torch
from mmdet.core.anchor import AnchorGenerator
from mmdet.core.anchor.builder import ANCHOR_GENERATORS

from sphdet.bbox.box_formator import Planar2SphBoxTransform


@ANCHOR_GENERATORS.register_module()
class SphAnchorGenerator(AnchorGenerator):
    """Spherical anchor generator for 2D anchor-based detectors.

    Horizontal bounding box represented by (theta, phi, alpha, beta).
    """
    def __init__(self, box_formator='sph2pix', box_version=4, *args, **kwargs):
        super(SphAnchorGenerator, self).__init__(*args, **kwargs)
        assert box_formator in ['sph2pix', 'pix2sph', 'sph2tan', 'tan2sph']
        assert box_version in [4, 5]
        self.box_formator = Planar2SphBoxTransform(box_formator, box_version)


    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        anchors = super(SphAnchorGenerator, self).single_level_grid_priors(featmap_size, level_idx, dtype, device)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        img_h, img_w = feat_h * stride_h, feat_w * stride_w

        sph_anchors = self.box_formator(anchors, (img_h, img_w))
        return sph_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        anchors = super(SphAnchorGenerator, self).single_level_grid_anchors(base_anchors, featmap_size, stride, device)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = stride
        img_h, img_w = feat_h * stride_h, feat_w * stride_w

        sph_anchors = self.box_formator(anchors, (img_h, img_w))
        return sph_anchors
       