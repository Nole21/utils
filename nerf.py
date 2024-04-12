from torch import Tensor
import torch
import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn
# 



def idx_to_3d(idx: int, H: int, W: int):
    """
    convert a 1D index to a 3D index(img_idx, row_idx, col_idx)
    Args:
        idx (int): The 1D index to convert
        H (int): The height of the 3D space
        W (int): The width of the 3D space
    Returns:
        tuple: A tuple containing the 3D index (i, j, k),
        where i is the image index,
        j is the row index,
        k is the colume index.
    """
    i = idx // (H * W)
    j = (idx % (H * W)) // W
    k = idx % W
    return i, j, k



def get_rays(
    x: Tensor, y: Tensor, c2w: Tensor, intrinsic: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    compute 3D rays direction and origin from 2D pixel index
    Args:
        x: the (column)horizontal coordinates of the pixels, shape: (num_rays,)
        y: the (row)vertical coordinates of the pixels, shape: (num_rays,)
        c2w: the camera-to-world matrices, shape: (num_rays, 4, 4)/(num_rays, 3, 3)
        intrinsic: the camera intrinsic matrices, shape: (num_rays, 3, 3)
    Rreturns:
        origins: the ray origins, shape: (num_rays, 3)
        viewdirs: the ray directions, shape: (num_rays, 3)
        direction_norm: the norm of the ray directions, shape: (num_rays, 1)
    """
    if len(intrinsic.shape) == 2:
        intrinsic = intrinsic[None, :, :]
    if len(c2w.shape) == 2:
        c2w = c2w[None, :, :]
    camera_dirs = torch.stack(
        [(x - intrinsic[:, 0, 2] + 0.5) / intrinsic[:, 0, 0],
         (y - intrinsic[:, 1, 2] + 0.5) / intrinsic[:, 1, 1],
         torch.ones_like(x).to(x)],
         dim=-1
    )

    
    # convert the camera directions from camera frame to world frame
    directions = ( camera_dirs[:, None, :] * c2w[:, :3, :3] ).sum(dim=-1)
    # directions1 = ( c2w[:, :3, :3] @ camera_dirs[:, :, None]).sum(dim=-1)
    # assert directions.equal(directions1),"matrix multiple error"
    direction_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
    viewdirs = directions / ( direction_norm + 1e-8 )
    origins = torch.broadcast_to(c2w[:,:3,-1], viewdirs.shape)
    return origins, viewdirs, direction_norm

    



class XYZ_Encoder(nn.Module):
    encoder_type = 'XYZ_Encoder'
    """Encode XYZ coordinates or directions to a vector"""

    def __init__(self,n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError



class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in NeRF"""

    def __init__(
            self,
            n_input_dims=3,
            min_deg: int = 0,
            max_deg: int = 10,
            enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        # buffer will be part of the parameters but will not be optimized
        self.register_buffer(
            "scales", Tensor([2.**i for i in range(min_deg, max_deg+1)])
        )


    @property
    def n_output_dims(self) -> int:
        return (int(self.enable_identity) + 2 * (self.max_deg - self.min_deg + 1)) * self.n_input_dims
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [...,n_input_dims]
        Returns:
            encoded:[..., n_output_dims]
        """
        if self.min_deg == self.max_deg:
            return x
        
        """
        # this implementation will be like (sin, sin, sin, ..., cos, cos, cos)
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded
        """

        # this implementation will be like (sin, cos, sin, cos, sin, cos,...)
        xb = x[...,None,:] * self.scales[:,None] # [..., max_deg-min_deg+1, input_dims]
        xb = torch.cat([xb, xb + 0.5*torch.pi], dim=-1).reshape(list(x.shape[:-1])+[2 * self.n_input_dims * (self.max_deg - self.min_deg + 1)])
        encoded = torch.sin(xb)
        if self.enable_identity:
            encoded = torch.cat([x, encoded], dim=-1)
        return encoded
   

# according to NerfAcc implementation
def render_transmittance_from_density(
        t_starts: Tensor,
        t_ends: Tensor,
        sigmas: Tensor,
):
    """
    Compute transmittance of each intervals or points along the ray
    Args:
        t_starts: the start distance of each interval along the ray, (num_rays, num_samples)
        t_ends: the end distance of each interval along the ray, (num_rays, num_samples)
        sigmas: predicted density for each point along the ray, (num_rays, num_samples)
    Returns:
        transmittance: transmittance of each point or interval along the ray, (num_rays, num_samples)
        alpha: alpha value of each point along the ray, (num_rays, num_samples)
    """
    sigmas = sigmas.squeeze(-1)
    sigmas_delta = sigmas * (t_ends - t_starts)
    alphas = 1 - torch.exp(sigmas_delta)
    trans = torch.exp( - torch.cumsum(
        torch.cat([torch.zeros_like(sigmas_delta.shape[..., :1]), sigmas_delta[..., :-1]], dim=-1),
        dim=-1
    ))
    return trans, alphas



# according to NeRF official pytorch implementation
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's output to semantically meaningful valuse.
    Args:
    raw: [num_rays, num_samples, 4]
    z_vals: [num_rays, num_samples along ray]
    rays_d: [num_rays, 3]
    Returns:
    rgb_map: [num_rays, 3]
    disp_map: [num_rays,] disparity map, inverse of depth map
    acc_map: [num_rays,]
    weights: [num_rays, num_samples]
    depth_map: [num_rays]. Estimated distance to object
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1) #[N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
    alpha = raw2alpha(raw[...,3]+noise, dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0],1)), 1.-alpha+1e-10], -1),-1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals ,-1)
    return rgb_map, weights, depth_map



def accumulate_along_rays(
        weights: Tensor,
        values: Tensor = None,
) -> Tensor:
    """
    Accumulate volumetric values along the ray like rgb or opacity
    Args:
        weights: the weights for accumulating each points along the ray, (num_rays, num_samples)
        values: the value for accumulation of each points, (num_rays, num_samples, D)
    Retures:
        output: the accumulated value for each ray, (num_rays, D)
    """
    if values is None: # used for accumulating opacity(density)
        src = weights[..., None]
    else:
        assert values.dim() == weights.dim() + 1
        assert values.shape[...,:-1] == weights.shape
        src = weights[..., None] * values
    output = torch.sum(src, dim=-2)
    return output



def compute_psnr(prediction, target):
    if not isinstance(prediction, Tensor):
        prediction = torch.tensor(prediction)
    if not isinstance(target, Tensor):
        target = torch.tensor(target)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def create_meshgrid(height, width):
    x = torch.linspace(0, width-1, width)
    y = torch.linspace(0, height-1, height)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((grid_x, grid_y), dim=-1)


