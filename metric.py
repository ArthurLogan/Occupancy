import torch


# calculate point cloud IoU
def IoU(model_out, gt):
    """[B, N] -> [B]"""
    intersection = ((model_out + gt) > 1).sum(dim=1)
    union = ((model_out + gt) > 0).sum(dim=1)
    iou = intersection / union
    return iou
