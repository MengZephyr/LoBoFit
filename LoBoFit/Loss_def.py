import torch

def compute_feats_loss(feats: torch.Tensor, gt_feats: torch.Tensor, weight: torch.Tensor=None):
    diff = feats - gt_feats
    feat_dim = feats.dim()
    loss = diff * diff
    if feat_dim > 1:
        loss = loss.sum(dim=-1)
    if weight is not None:
        loss = weight * loss

    return loss.sum()

def compute_chain_feats_loss(c_feats, gt_c_feats, num_c, c_weights):
    assert len(c_feats) == num_c and len(gt_c_feats) == num_c
    if c_weights is not None:
        assert len(c_weights) == num_c

    total_loss = 0.0
    for i in range(num_c):
        f = c_feats[i]
        gt_f = gt_c_feats[i]
        w = None
        if c_weights is not None:
            w = c_weights[i]
        loss = compute_feats_loss(f, gt_f, w)
        total_loss = total_loss + loss.sum()

    return total_loss


def compute_reg_feats(feats, method = 'mean'):
    loss = (feats * feats).sum(dim=-1)
    if method == 'mean':
        loss = loss.mean()
    else:
        loss = loss.sum()
    return loss


def compute_loss_collision(base_verts, base_norms, verts, d_thre=0.001, wei=None, if_gate=False):

    diff = verts - base_verts
    proj = diff * base_norms
    d = proj.sum(dim=-1)  # [N]
    cc = torch.relu(d_thre - d)

    if wei is not None:
        if if_gate:
            gate = (cc > 0).to(cc.dtype)
            wei = wei * gate
        sw = torch.sum(wei).clamp_min(1e-12)
        loss = torch.sum(wei * cc) / sw
    else:
        loss = torch.sum(cc)
    return loss, d


def reg_delta_kw(delta_abkw):
    delta_k = delta_abkw[:, :, 2]   # [M, J]
    delta_w = delta_abkw[:, :, -1]  # [M, J]
    loss_k = torch.sum(delta_k*delta_k)
    loss_w = torch.sum(delta_w*delta_w)
    return loss_k, loss_w




