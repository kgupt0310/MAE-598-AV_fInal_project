import torch


def chamfer_distance(pred, gt):
    dist = torch.cdist(pred, gt)
    min_pred_to_gt = dist.min(dim=2)[0]
    min_gt_to_pred = dist.min(dim=1)[0]
    return min_pred_to_gt.mean() + min_gt_to_pred.mean()


def repulsion_loss(points, radius=0.03):
    B, N, _ = points.shape
    dist = torch.cdist(points, points)

    eye = torch.eye(N, device=points.device).unsqueeze(0)
    dist = dist + eye * 1e6

    penalty = torch.clamp(radius - dist, min=0.0)
    return penalty.mean()


def reconstruction_loss(pred, clean, corrupted, attn_weights=None):
    chamfer = chamfer_distance(pred, clean)

    pointwise = torch.norm(pred - clean, dim=2)
    l2 = pointwise.mean()

    repel = repulsion_loss(pred)
    stability = torch.norm(pred - corrupted, dim=2).mean()

    attn_term = torch.tensor(0.0, device=pred.device)
    if attn_weights is not None:
        attn_term = (attn_weights * pointwise).mean()

    loss = (
        chamfer
        + 0.05 * l2
        + 0.05 * repel
        + 0.01 * stability
        + 0.10 * attn_term
    )

    parts = {
        "chamfer": chamfer.item(),
        "l2": l2.item(),
        "repulsion": repel.item(),
        "stability": stability.item(),
        "attn": attn_term.item(),
    }

    return loss, parts
