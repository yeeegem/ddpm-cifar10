import argparse
import math
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import datasets, transforms

from diffusion import Diffusion
from model import UNet

T = 1000
N_SAMPLES = 64
GIF_INTERVAL = 100  # ms per frame


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


def save_comparison(real, generated, path, nrow=8):
    """Real CIFAR-10 (top row) vs generated (bottom row)."""
    fig, axes = plt.subplots(2, nrow, figsize=(nrow * 1.5, 3.5))
    for col in range(nrow):
        axes[0, col].imshow(to_img(real[col]))
        axes[0, col].axis("off")
        axes[1, col].imshow(to_img(generated[col]))
        axes[1, col].axis("off")
    fig.text(0.01, 0.75, "real", va="center", rotation="vertical", fontsize=9)
    fig.text(0.01, 0.25, "generated", va="center", rotation="vertical", fontsize=9)
    plt.tight_layout(rect=[0.03, 0, 1, 1], pad=0.2)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


def save_gif(frames, path, interval=GIF_INTERVAL):
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


def load_real_samples(n, data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    indices = torch.randperm(len(dataset))[:n]
    return torch.stack([dataset[i][0] for i in indices])


@torch.no_grad()
def sample_with_frames(diffusion, model, device, seed=0):
    """Reverse chain for one image. Sparse in the noise phase, dense in the
    last 200 steps where visible structure forms."""
    torch.manual_seed(seed)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)
    frames = []
    for t in reversed(range(diffusion.T)):
        x = diffusion.p_sample(model, x, t)
        if t >= 200 and t % 50 == 0:   # ~16 frames, noise phase
            frames.append(x[0].clone())
        elif t < 200 and t % 4 == 0:   # 50 frames, formation phase
            frames.append(x[0].clone())
    model.train()
    # frames are already in noisy->clean order (loop runs t=999 down to 0)
    return x, frames


def epoch_from_checkpoint(path):
    """Extract epoch number from filenames like ckpt_epoch0120.pt."""
    name = os.path.splitext(os.path.basename(path))[0]
    for part in name.split("_"):
        if part.startswith("epoch") and part[5:].isdigit():
            return int(part[5:])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="samples_output")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--gif_seed", type=int, default=0,
                        help="seed for the GIF sample; change to get a different image")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    epoch = epoch_from_checkpoint(args.checkpoint)
    prefix = f"epoch{epoch:04d}_"

    png_path = os.path.join(args.out_dir, f"{prefix}samples.png")
    cmp_path = os.path.join(args.out_dir, f"{prefix}comparison.png")
    gif_path = os.path.join(args.out_dir, f"{prefix}denoising.gif")

    existing = os.listdir(args.out_dir)
    already_done = [f for f in existing if f.startswith(prefix)]
    if already_done:
        print(f"already have outputs for epoch {epoch}: {already_done}")
        print("delete them or pick a different checkpoint to regenerate")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()

    diffusion = Diffusion(T=T, device=device)

    samples = diffusion.sample(model, N_SAMPLES, device)
    save_grid(samples, png_path)

    real = load_real_samples(8, args.data_dir)
    save_comparison(real, samples[:8], cmp_path)

    _, frames = sample_with_frames(diffusion, model, device, seed=args.gif_seed)
    save_gif(frames, gif_path)


if __name__ == "__main__":
    main()
