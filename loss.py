import torch
import torch.nn.functional as F


def flow_loss_func(flow_preds, flow_gt, valid,
                   gamma=0.9,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # [B, H, W]
    valid = (valid >= 0.5)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_gt).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'AEE': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics
def flow_loss_func_unsupervised(flow_preds, img0, img1, gamma=0.9, q = 0.1, h = 0.9, alpha = 0.8, beta = 0.2):
    _, _, H, W = img0.shape
    removal_radius = 4
    weight_scaling = 3.439
    
    valid_slice = (
        slice(None),
        slice(None),
        slice(removal_radius-1, H-removal_radius+1),
        slice(removal_radius-1, W-removal_radius+1)
    )
 
    gray_loss = torch.tensor(0.0, device=img0.device)
    MrDGC_loss = torch.tensor(0.0, device=img0.device)

    for scale_idx in range(len(flow_preds)):

        scale_weight = (gamma ** (len(flow_preds)-scale_idx-1)) / weight_scaling
        
        warped_img = warp(img1, flow_preds[scale_idx])
        
        gray_diff = ((img0[valid_slice] - warped_img[valid_slice]).abs()) / 255.0

        gray_loss += scale_weight * gray_diff.mean()

    MrDGC_loss = h * MrDGC(flow_preds[-1], flow_preds[2]) + q * MrDGC(flow_preds[-1], flow_preds[0])

    total_loss = alpha * gray_loss + beta * MrDGC_loss

    metrics = {
        'Gray': gray_loss.item(),
        'MrDGC': MrDGC_loss.item(),
        'Total': total_loss.item()
    }
    
    return total_loss, metrics

def gradient_abs_mean(tensor):

    grad_x = torch.gradient(tensor, dim=1, edge_order=1)[0]
    grad_y = torch.gradient(tensor, dim=2, edge_order=1)[0]

    grid = torch.stack([grad_x, grad_y], dim=1)
    return grid

def smooth_L1(flow):
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
        
    reg_u = gradient_abs_mean(u)
    reg_v = gradient_abs_mean(v)
        
    return reg_u, reg_v

def MrDGC(flow1, flow2):

    K = 20
    
    gu_1, gv_1 = smooth_L1(flow1)
    gu_2, gv_2 = smooth_L1(flow2)

    C0 = 1e-6

    diff_u = torch.sum(((gu_1 - gu_2).abs()), dim = 1)
    diff_v = torch.sum(((gv_1 - gv_2).abs()), dim = 1)

    diff_u = (diff_u / K).mean()
    diff_v = (diff_v / K).mean()

    return (diff_u + diff_v) / 2

def warp(img, flow):
    height, width = img.shape[-2:]

    yy, xx = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
    xi1 = xx.to(flow.device) + flow[:,0,:,:]
    yi1 = yy.to(flow.device) + flow[:,1,:,:]

    grid1 = torch.stack((xi1/(width-1)*2-1, 
                            yi1/(height-1)*2-1),-1)
    output = F.grid_sample(img, grid1, mode='bicubic',align_corners=True, padding_mode='zeros')
    
    return output