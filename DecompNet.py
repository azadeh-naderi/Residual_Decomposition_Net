# Lightweight Decomposer Networks with Mini CNN Autoencoders + FIXED SPATIAL MASKS
# - 3-layer CNNs, smaller feature maps (56x46) for laptop speed
# - N fixed Gaussian masks applied before each AE (residual is masked)
# - Masks: random centers, radius chosen so 0.5-level set covers half image area
# Requires: pip install torch pillow numpy matplotlib

import os, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms








# ----------------------------
# Dataset loaders
# ----------------------------
class ORLFaces(Dataset):
    """
    ORL faces loader expecting .pgm files under: root/s1/*.pgm, root/s2/*.pgm, ...
    Returns tensor: (1, H, W)
    """
    def __init__(self, root: str, H: int = 56, W: int = 46, seed: int = 0):
        self.H, self.W = H, W
        self.paths = sorted(glob.glob(os.path.join(root, "s*/*.pgm")))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No .pgm found under {root} (expected root/s*/**.pgm)")

        rng = np.random.default_rng(seed)
        _ = rng  # kept for parity; not used currently

        X = []
        for p in self.paths:
            img = Image.open(p).convert("L").resize((self.W, self.H))
            X.append(np.asarray(img, dtype=np.float32).reshape(-1))
        X = np.stack(X, axis=0)

        # Keep your original ORL normalization style (z-score per pixel)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        self._X = X

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self._X[i:i+1, :]
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.astype(np.float32)).squeeze(0)
        return x.view(1, self.H, self.W)


class TorchvisionWrapper(Dataset):
    """
    Wrap torchvision datasets so __getitem__ returns only the image tensor.
    """
    def __init__(self, tv_ds):
        self.ds = tv_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        return x


def build_dataset(dataset_name: str, root: str, train: bool = True):
    """
    Returns:
      ds: Dataset producing x in shape (C,H,W)
      H, W: int
      C: int
    """
    name = dataset_name.strip().lower()

    if name == "orl":
        ds = ORLFaces(root=root, H=56, W=46, seed=0)
        return ds, 56, 46, 1

    if name in ("cifar-10", "cifar10", "cifar_10"):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        tv = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
        return TorchvisionWrapper(tv), 32, 32, 3

    if name in ("cifar-100", "cifar100", "cifar_100"):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        tv = datasets.CIFAR100(root=root, train=train, download=True, transform=tfm)
        return TorchvisionWrapper(tv), 32, 32, 3

    if name in ("imagenet", "image-net", "ImageNet"):
        # Assumes ImageNet is already downloaded and arranged as:
        #   root/train/<class>/*.jpeg
        #   root/val/<class>/*.jpeg
        split = "train" if train else "val"
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f"ImageNet split folder not found: {split_root}\n"
                f"Expected structure: root/train/<class>/..., root/val/<class>/..."
            )

        # Keep it simple + consistent masks: fixed 224x224
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        tv = datasets.ImageFolder(root=split_root, transform=tfm)
        return TorchvisionWrapper(tv), 224, 224, 3

    raise ValueError(f"Unknown dataset_name='{dataset_name}'. Use: orl, cifar10, cifar100, imagenet")

# ----------------------------
# Spatial Gaussian mask bank
# ----------------------------
def make_gaussian_mask(H: int, W: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """
    2D Gaussian centered at (cx, cy) in pixel coords (x=cols in [0..W-1], y=rows in [0..H-1])
    Returns array in [0,1], peak ~1 at center.
    """
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2.0 * sigma**2))
    g = g.astype(np.float32)
    return g
    
    
class SpatialMasks:
    """
    Build N fixed Gaussian masks for (H,W) images.
    Centers random once; sigma chosen s.t. 0.5-level set covers half the image area.
    """
    def __init__(self, H: int, W: int, N: int, seed: int = 42, to_torch_device: str = "cpu"):
        self.H, self.W, self.N = H, W, N
        rng = np.random.default_rng(seed)

        # radius so that circle area = 0.5 * H*W
        R = math.sqrt((0.2 * H * W) / math.pi)
        # choose sigma so that G(R) = 0.5 => exp(-R^2/(2sigma^2)) = 0.5
        sigma = R / math.sqrt(2.0 * math.log(2.0))
        sigmas = [1*sigma,0.4*sigma,0.4*sigma,0.8*sigma,0.9*sigma]
        masks = []
        #mask_locs_x = [np.round(W / 2), np.round(1.25*W / 4), np.round(2.75*W / 4), np.round(W / 2), np.round(W / 2)]
        #mask_locs_y = [0, np.round(4.4*H / 9), np.round(4.4*H / 9), np.round(6*H / 9), H]
        for n in range(N):
            cx = rng.uniform(0, W-1)  # random center across width
            cy = rng.uniform(0, H-1)  # random center across height
            #cx=mask_locs_x[n]
            #cy=mask_locs_y[n]
            g = make_gaussian_mask(H, W, cx, cy, sigmas[n])
            g = g / (g.max() + 1e-8)  # normalize peak to 1
            masks.append(g[None, None, :, :])  # shape (1,1,H,W)
        self.masks_np = np.stack(masks, axis=0)  # (N,1,1,H,W) but we used (1,1,H,W) appended -> actually (N,1,1,H,W)
        self.masks_np = self.masks_np.squeeze(2) # -> (N,1,H,W)
        self.device = to_torch_device
        self._to_torch()

    def _to_torch(self):
        self.masks = torch.from_numpy(self.masks_np.copy()).to(self.device)  # (N,1,H,W)
        self.masks.requires_grad_(False)

    def to(self, device: torch.device):
        self.device = device
        self.masks = self.masks.to(device)
        return self

    def get(self, i: int) -> torch.Tensor:
        return self.masks[i]  # (1,H,W)

    def show(self, cols: int = 5, title: str = "Fixed Gaussian Masks"):
        N = self.N
        rows = int(np.ceil(N / cols))
        plt.figure(figsize=(2.0*cols, 2.2*rows))
        for i in range(N):
            plt.subplot(rows, cols, i+1)
            plt.imshow(self.masks_np[i,0], cmap='magma')
            plt.axis('off')
            plt.title(f"M_{i+1}")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# ----------------------------
# Mini 3-layer CNN Autoencoder
# ----------------------------


class MiniCNNAE(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        c_in = int(input_channels)
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, c_in, 3, padding=1), nn.Tanh(),
        )

    def forward(self, r):
        return self.dec(self.enc(r))


# ----------------------------
# Core residual & training logic (with masks)
# ----------------------------


def damp_update(prev, now, alpha=0.5):
    return (1 - alpha) * prev + alpha * now


def residual_sweeps(models, masks: SpatialMasks, x, sigma, K=2, alpha=0.5):
    """
    x: (B,C,H,W), sigma: (B,N), masks: (N,1,H,W)
    Each residual r_i is multiplied by mask_i before AE_i.
    """
    B, C, H, W = x.shape
    N = len(models)
    comps = [torch.zeros_like(x) for _ in range(N)]

    for _ in range(K):
        for i in range(N):
            recon_others = torch.sum(
                torch.stack(
                    [sigma[:, j:j+1].view(B, 1, 1, 1) * comps[j] for j in range(N) if j != i],
                    dim=0
                ),
                dim=0
            )
            r_i = x - recon_others
            # apply fixed spatial mask_i
            mask_i = masks.get(i).view(1, 1, H, W).to(x.device)  # (1,1,H,W)
            r_i_masked = r_i * mask_i  # broadcast over C
            xhat_i = models[i](r_i_masked)
            comps[i] = damp_update(comps[i], xhat_i, alpha)

    return comps
    

# no-grad sweeps (used only for sigma solving)
@torch.no_grad()
def residual_sweeps_nograd(models, masks: SpatialMasks, x, sigma, K=2, alpha=0.5):
    B, C, H, W = x.shape
    N = len(models)
    comps = [torch.zeros_like(x) for _ in range(N)]

    for _ in range(K):
        for i in range(N):
            recon_others = torch.sum(
                torch.stack(
                    [sigma[:, j:j+1].view(B, 1, 1, 1) * comps[j] for j in range(N) if j != i],
                    dim=0
                ),
                dim=0
            )
            r_i = x - recon_others
            mask_i = masks.get(i).view(1, 1, H, W).to(x.device)
            r_i_masked = r_i * mask_i
            xhat_i = models[i](r_i_masked)
            comps[i] = damp_update(comps[i], xhat_i, alpha)

    return comps
  
  
def solve_sigma_ridge(x, comps, eps=1e-4):

    B, C, H, W = x.shape
    d = C * H * W
    N = len(comps)

    X = x.view(B, d)
    Hs = torch.stack([c.view(B, d) for c in comps], dim=2)  # (B,d,N)

    A = Hs.transpose(1, 2) @ Hs + eps * torch.eye(N, device=x.device).unsqueeze(0)  # (B,N,N)
    b = (Hs.transpose(1, 2) @ X.unsqueeze(2)).squeeze(2)  # (B,N)

    sigma = torch.linalg.solve(A, b)
    return torch.clamp(sigma, min=0)
    
    
    
def loss_reconstruction(x, comps, sigma):
    recon = torch.sum(
        torch.stack([sigma[:, i:i+1].view(-1, 1, 1, 1) * comps[i] for i in range(len(comps))], dim=0),
        dim=0
    )
    return F.mse_loss(recon, x)
    
    
    
def train_step(models, masks: SpatialMasks, x, opt, step, K=2, alpha=0.5, eps_sigma=1e-4, warmup=5):
    for m in models:
        m.train()

    B = x.shape[0]
    N = len(models)
    device = x.device

    # ---- sigma (no-grad) ----
    with torch.no_grad():
        sigma = torch.ones(B, N, device=device)
        if step >= warmup:
            comps_ng = residual_sweeps_nograd(models, masks, x, sigma, K, alpha)
            sigma = solve_sigma_ridge(x, comps_ng, eps_sigma)

    # ---- backprop pass (grad-enabled sweeps) ----
    opt.zero_grad(set_to_none=True)
    comps_tmp = residual_sweeps(models, masks, x, sigma, K, alpha)
    rec = loss_reconstruction(x, comps_tmp, sigma.detach())
    rec.backward()
    torch.nn.utils.clip_grad_norm_([p for m in models for p in m.parameters()], 3.0)
    opt.step()

    return float(rec.item())


# ----------------------------
# Visualization
# ----------------------------


@torch.no_grad()
def show_parts(models, masks: SpatialMasks, ds, idx=idx, K=2, eps=1e-4, save_path=None):
    device = next(models[0].parameters()).device
    x = ds[idx].unsqueeze(0).to(device)  # (1,C,H,W)

    N = len(models)
    sigma = torch.ones(1, N, device=device)
    comps = residual_sweeps_nograd(models, masks, x, sigma, K=K, alpha=0.5)
    sigma = solve_sigma_ridge(x, comps, eps)
    comps = residual_sweeps_nograd(models, masks, x, sigma, K=K, alpha=0.5)

    parts = [(sigma[:, i:i+1].view(1, 1, 1, 1) * comps[i]).squeeze(0).cpu().numpy() for i in range(N)]
    recon = np.sum(parts, axis=0)
    ximg = x.squeeze(0).cpu().numpy()

    C = ximg.shape[0]
    cols = N + 2
    plt.figure(figsize=(2.2 * cols, 2.6))

    if C == 1:
        # grayscale
        plt.subplot(1, cols, 1)
        plt.imshow(ximg[0], cmap="gray")
        plt.axis("off")
        plt.title("x")

        for i, p in enumerate(parts, 1):
            plt.subplot(1, cols, 1 + i)
            plt.imshow(p[0], cmap="gray")
            plt.axis("off")
            plt.title(f"σ{i} x̂{i}")

        plt.subplot(1, cols, cols)
        plt.imshow(recon[0], cmap="gray")
        plt.axis("off")
        plt.title("sum = x̂")

    else:
        # RGB (assumes normalized roughly to [-1,1] via mean=0.5/std=0.5 like CIFAR/ImageNet)
        def to_img(arr_chw):
            arr_chw = np.clip(arr_chw, -1, 1)
            return arr_chw.transpose(1, 2, 0) * 0.5 + 0.5

        plt.subplot(1, cols, 1)
        plt.imshow(to_img(ximg))
        plt.axis("off")
        plt.title("x")

        for i, p in enumerate(parts, 1):
            plt.subplot(1, cols, 1 + i)
            plt.imshow(to_img(p))
            plt.axis("off")
            plt.title(f"σ{i} x̂{i}")

        plt.subplot(1, cols, cols)
        plt.imshow(to_img(recon))
        plt.axis("off")
        plt.title("sum = x̂")

    plt.tight_layout()
    plt.savefig("output.png")
    plt.show()
    
@torch.no_grad()
def show_masks(masks: SpatialMasks, cols: int = 5):
    masks.show(cols=cols, title="Fixed Gaussian Masks (0.5-level ~ half image area)")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # ---- user knobs ----
    dataset_name = "cifar10"  # "orl", "cifar10", "cifar100", "imagenet"
    root = "./data"           # ORL root or CIFAR download dir or ImageNet root
    train_split = True

    N = 5          # number of subnetworks/masks
    batch_size = 32
    steps = 50     # "epochs" over the loader in your original style
    lr = 1e-3

    idx = 1000

    K = 2
    alpha = 0.5
    warmup = 5
    eps_sigma = 1e-4

    # ---- setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, H, W, input_channels = build_dataset(dataset_name, root=root, train=train_split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    masks = SpatialMasks(H, W, N, seed=123).to(device)
    show_masks(masks, cols=N)

    models = [MiniCNNAE(input_channels=input_channels).to(device) for _ in range(N)]
    opt = torch.optim.Adam([p for m in models for p in m.parameters()], lr=lr)

    # ---- train ----
    for step in range(steps):
        for batch in loader:
            x = batch.to(device)  # (B,C,H,W)
            loss = train_step(
                models, masks, x, opt, step,
                K=K, alpha=alpha, eps_sigma=eps_sigma, warmup=warmup
            )
        if (step + 1) % 5 == 0:
            print(f"[{step+1:03d}] loss={loss:.4f}")

    # ---- visualize ----
    show_parts(models, masks, ds, idx=idx, K=K, eps=eps_sigma, save_path=None)