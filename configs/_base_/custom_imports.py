custom_imports = dict(
    imports=[
        'sphdet.datasets',
        'sphdet.datasets.pipelines',
        'sphdet.models.detectors',
        'sphdet.models.heads',
        'sphdet.bbox.anchor',
        'sphdet.bbox.coder',
        'sphdet.bbox.sampler',
        'sphdet.bbox.nms',
        'sphdet.iou',
        'sphdet.losses'
    ],
    allow_failed_imports=False
)