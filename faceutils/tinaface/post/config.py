# 2. model
use_sigmoid = True
num_classes = 1
strides = [4, 8, 16, 32, 64, 128]
scales_per_octave = 3
ratios = [1.3]
input_size = 800

# 3. engine
meshgrid = dict(
    typename='BBoxAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='BBoxBaseAnchor',
        octave_base_scale=2**(4 / 3),
        scales_per_octave=scales_per_octave,
        ratios=ratios,
        base_sizes=strides))

bbox_coder = dict(
    typename='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[0.1, 0.1, 0.2, 0.2])

converter = dict(
    typename='IoUBBoxAnchorConverter',
    num_classes=num_classes,
    bbox_coder=bbox_coder,
    nms_pre=-1,
    use_sigmoid=use_sigmoid)

test_cfg = dict(
    min_bbox_size=0,
    score_thr=0.01,
    iou_thr=0.45,
    max_per_img=-1)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=False)


transforms = [
    dict(typename='LoadImageFromFile'),
    dict(typename='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(typename='Normalize', **img_norm_cfg),
    dict(typename='Pad', size=(800, 800), size_divisor=None, pad_val=0, around=False),
    dict(typename='ImageToTensor', keys=['img']),
    dict(typename='Collect', keys=['img'], meta_keys=('ori_shape',
                                                      'img_shape', 'pad_shape', 'scale_factor',
                                                      'img_norm_cfg'))
]


def get_kwargs(param: dict, typename=''):
    kwargs = dict()
    for key, val in param.items():
        if key == 'typename':
            assert val == typename
        else:
            kwargs[key] = val
    return kwargs
