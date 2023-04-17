import torch
import numpy as np


def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = torch.stack([torch.histc(image[c], bins=256, min=0, max=255) for c in range(3)])
    hists = [hist / torch.sum(hist) for hist in hists]
    hists = [torch.cumsum(hist, dim=0) for hist in hists]
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    front_adj = adj[: 255]
    after_adj = adj[1:]

    ref_com = ref[1:]

    front_adj_tensor = front_adj[None, :].repeat(255, 1)
    after_adj_tensor = after_adj[None, :].repeat(255, 1)
    ref_tensor = ref_com[:, None].repeat(1, 255)

    front_table = ref_tensor - front_adj_tensor
    after_table = ref_tensor - after_adj_tensor

    front_table = torch.where(front_table >= 0, 1., 0.)
    after_table = torch.where(after_table <= 0, 1., 0.)

    table_index = torch.nonzero(front_table * after_table).cpu().numpy().tolist()
    table_group = dict()
    for idv in table_index:
        if table_group.get(idv[0], None) is not None:
            continue
        table_group[idv[0]] = idv[1]

    table = [i for i in range(0, 256)]
    for i, j in table_group.items():
        table[i + 1] = j + 1
    table[255] = 255

    return torch.tensor(np.array(table)).to(ref.device)


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """

    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = tables[dst_align[i].long()].float()

    return dstImg


def cal_hist_channel(image):
    """
        cal cumulative hist for channel list
    """
    hists = torch.histc(image, bins=256, min=0, max=255)
    hists = hists / torch.sum(hists)
    hists = torch.cumsum(hists, dim=0)
    return hists


def histogram_matching_channel(dstImg, refImg, index):
    dst_align = dstImg[index[0], index[1]]
    ref_align = refImg[index[2], index[3]]

    hist_ref = cal_hist_channel(ref_align)
    hist_dst = cal_hist_channel(dst_align)

    tables = cal_trans(hist_dst, hist_ref)

    dstImg[index[0], index[1]] = tables[dst_align.long()].float()
    return dstImg
