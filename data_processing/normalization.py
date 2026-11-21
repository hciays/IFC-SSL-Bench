import torch
from flowutils.transforms import logicle 
from torch.nn import functional as F, Module


def pad_img_to_multiple(x, multiple=16):  # for vit patches
    # x: [C, H, W], with H = W
    C, H, W = x.shape
    NH = ((H + multiple - 1) // multiple) * multiple

    t = (NH - H) // 2
    b = NH - H - t
    if t == b == 0:
        return x
    # order: (left, right, top, bottom); replicate - channel-first
    return F.pad(x, (t, b, t, b), mode="replicate")


class PerImageNormalize:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-6  # to avoid division by zero
        return (x - mean) / std


class LogicleTransform(Module):
    """
    CHW-only. Assumes input in [0,1], maps to counts via T before calling flowutils.logicle.
    Defaults: 
        T = 2**18 = 262144
        M=4.5
        W=0.5
        A=0.0
    """
    # T = 2^16
    def __init__(self, T: float = 262144.0, M: float = 4.5, W: float = 0.5, A: float = 0.0):
        super().__init__()
        self.T = T
        self.M = M
        self.W = W
        self.A = A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W], float32
        C, H, W = x.shape
        
        arr = x.cpu().numpy().transpose(1, 2, 0).reshape(-1, C)  # (H*W, C)
        arr_t = logicle(arr, None, t=self.T, m=self.M, w=self.W, a=self.A)
        
        out = torch.from_numpy(arr_t.reshape(H, W, C).transpose(2, 0, 1))  # back to CHW
        return out.to(x.device, dtype=torch.float32)
