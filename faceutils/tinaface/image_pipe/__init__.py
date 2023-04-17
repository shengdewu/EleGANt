from .transformer import *


transformer_type = {
    'LoadImageFromFile': LoadImageFromFile,
    'Resize': Resize,
    'Normalize': Normalize,
    'Pad': Pad,
    'ImageToTensor': ImageToTensor,
    'Collect': Collect,
}