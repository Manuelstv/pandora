_base_ = [
    '../_base_/custom_imports.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_120e.py',
    '../_base_/datasets/indoor360.py',
    '../_base_/models/sph_retinanet_r50_fpn.py',
]

# log
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5)

optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=5, norm_type=2))

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2)


model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            box_formator='sph2pix'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='KentLoss', loss_weight = 30., img_size = (128,256))),
    train_cfg=dict(
        assigner=dict(
            iou_calculator=dict(
                backend='sph2pob_standard_iou')),),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        iou_calculator='unbiased_iou'))