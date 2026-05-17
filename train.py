import copy
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusion import Diffusion
from model import UNet

T = 1000
BATCH_SIZE = 128
LR = 2e-4
N_EPOCHS = 500
EMA_DECAY = 0.9999
SAVE_EVERY = 50
LOG_EVERY = 100
CHECKPOINT_DIR = "checkpoints"


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1 - decay)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    model = UNet().to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    diffusion = Diffusion(T=T, device=device)
    opt = optim.Adam(model.parameters(), lr=LR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    start_epoch = 1
    total_steps = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from {args.resume}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        epoch_loss = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device)  # imgs is x0 in the diffusion math, i.e. the clean image
            loss = diffusion.loss(model, imgs)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema_model, model, EMA_DECAY)

            epoch_loss += loss.item()
            total_steps += 1
            if total_steps % LOG_EVERY == 0:
                print(f"epoch {epoch:>4}  step {total_steps:>7}  loss {loss.item():.4f}")

        if epoch % SAVE_EVERY == 0:
            path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": opt.state_dict(),
            }, path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
