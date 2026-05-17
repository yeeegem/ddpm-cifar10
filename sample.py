import argparse
import math
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from diffusion import Diffusion
from model import UNet

T = 1000
N_SAMPLES = 64


def to_img(x):
    """Unnormalize from [-1,1] to [0,1] and clamp."""
    return ((x + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def save_grid(samples, path, nrow=8):
    n = samples.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 1.5, ncol * 1.5))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(to_img(samples[i]))
        ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def save_gif(frames, path, interval=60):
    """frames: list of (3,32,32) tensors at every 100th reverse step."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    imgs = []
    for f in frames:
        im = ax.imshow(to_img(f), animated=True)
        imgs.append([im])
    ani = animation.ArtistAnimation(fig, imgs, interval=interval, blit=True)
    ani.save(path, writer="pillow")
    plt.close(fig)
    print(f"saved {path}")


@torch.no_grad()
def sample_with_frames(diffusion, model, device):
    """Run reverse chain for one image, collecting frames every 100 steps."""
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    frames = []
    for t in reversed(range(diffusion.T)):
        x = diffusion.p_sample(model, x, t)
        if t % 100 == 0:
            frames.append(x[0].clone())
    model.train()
    return x, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()

    diffusion = Diffusion(T=T, device=device)

    samples = diffusion.sample(model, N_SAMPLES, device)
    save_grid(samples, os.path.join(args.out_dir, "samples.png"))

    _, frames = sample_with_frames(diffusion, model, device)
    frames.reverse()  # show noisy->clean in the GIF
    save_gif(frames, os.path.join(args.out_dir, "denoising.gif"))


if __name__ == "__main__":
    main()
