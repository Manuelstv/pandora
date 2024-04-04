_base_ = [
    './sph_retinanet_r50_fpn_120e_indoor360.py',
]

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_cls=dict(
            loss_weight=1.0),
        loss_bbox=dict(
            _delete_=True,
            type='Sph2PobIoULoss',
            mode='ciou',
            loss_weight=1.0)))