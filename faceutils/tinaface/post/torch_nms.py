import torch
import torchvision


def _batched_nms(bboxes, scores, inds, iou_thr, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        bboxes (torch.Tensor): bboxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        inds (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different inds,
            shape (N, ).
        iou_thr (float): specify nms type and class_agnostic as well as other
            parameters like iou_thr.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all bboxes,
            regardless of the predicted class

    Returns:
        tuple: kept bboxes and indice.
    """
    if class_agnostic:
        bboxes_for_nms = bboxes
    else:
        max_coordinate = bboxes.max()
        offsets = inds.to(bboxes) * (max_coordinate + 1)
        bboxes_for_nms = bboxes + offsets[:, None]

    keep = torchvision.ops.nms(boxes=bboxes_for_nms, scores=scores, iou_threshold=iou_thr)

    return keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    inds = scores.argsort(descending=True)
    bboxes = bboxes[inds]
    scores = scores[inds]
    labels = labels[inds]

    batch_bboxes = torch.empty((0, 4),
                               dtype=bboxes.dtype,
                               device=bboxes.device)
    batch_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
    batch_labels = torch.empty((0,), dtype=labels.dtype, device=labels.device)
    while bboxes.shape[0] > 0:
        num = min(100000, bboxes.shape[0])
        batch_bboxes = torch.cat([batch_bboxes, bboxes[:num]])
        batch_scores = torch.cat([batch_scores, scores[:num]])
        batch_labels = torch.cat([batch_labels, labels[:num]])
        bboxes = bboxes[num:]
        scores = scores[num:]
        labels = labels[num:]

        keep = _batched_nms(batch_bboxes, batch_scores, batch_labels, iou_thr)
        batch_bboxes = batch_bboxes[keep]
        batch_scores = batch_scores[keep]
        batch_labels = batch_labels[keep]

    dets = torch.cat([batch_bboxes, batch_scores[:, None]], dim=-1)
    labels = batch_labels

    if max_num > 0:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels
