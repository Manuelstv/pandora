import copy
import torch
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads import RPNHead

from sphdet.bbox.nms import PlanarNMS, SphNMS
import torch.nn as nn

@HEADS.register_module()
class SphRPNHead(RPNHead):
    def __init__(self, box_version=4, *args, **kwargs):
        assert box_version in [4, 5]
        self.box_version = box_version
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * self.box_version, 1)

    def _get_bboxes_single(self,
                       cls_score_list,
                       bbox_pred_list,
                       score_factor_list,
                       mlvl_anchors,
                       img_meta,
                       cfg,
                       rescale=False,
                       with_nms=True,
                       **kwargs):
        """Transform outputs of a single image into bbox predictions.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.box_version)
            
            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))
            
        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape)
    
    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            # w = proposals[:, 2] - proposals[:, 0]
            # h = proposals[:, 3] - proposals[:, 1]
            w = proposals[:, 2] / 360 * img_shape[1]
            h = proposals[:, 3] / 180 * img_shape[0]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            if cfg.iou_calculator == 'planar':
                nms = PlanarNMS(box_formator=cfg.box_formator)
            else:
                nms = SphNMS(iou_calculator=cfg.iou_calculator)
            dets, _ = nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, self.box_version+1)

        #dets_ = _sph2pix_box_transform(dets[:cfg.max_per_img, :-1], img_shape[:2])
        #dets = torch.torch.concat([dets_, dets[:cfg.max_per_img, -1][:, None]], dim=1)
        return dets[:cfg.max_per_img]

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, self.box_version)
        bbox_weights = bbox_weights.reshape(-1, self.box_version)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.box_version)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, self.box_version)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox