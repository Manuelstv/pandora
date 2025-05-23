import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.core.utils import filter_scores_and_topk, multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads import SSDHead

from sphdet.bbox.nms import PlanarNMS, SphNMS


@HEADS.register_module()
class SphSSDHead(SSDHead):
    def __init__(self, box_version=4, 
                #loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                *args, **kwargs):
        assert box_version in [4, 5]
        self.box_version = box_version
        super().__init__(*args, **kwargs)
        #self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.box_version,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))
    
    def loss_single(self, cls_score, bbox_pred, anchor, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        #loss_cls_all = self.loss_cls(
        #    cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        # loss_bbox = smooth_l1_loss(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     beta=self.train_cfg.smoothl1_beta,
        #     avg_factor=num_total_samples)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, self.box_version)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, self.box_version)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, self.box_version)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        #if rescale:
        #    mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            if cfg.iou_calculator == 'planar':
                nms = PlanarNMS(box_formator=cfg.box_formator)
            else:
                nms = SphNMS(iou_calculator=cfg.iou_calculator)
            det_bboxes, keep_idxs = nms(mlvl_bboxes, mlvl_scores,
                                        mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.box_version)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)