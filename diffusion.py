import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0, t, eps=None):
        # x0: clean image, x_t: noisy version at timestep t
        """Forward process: x_t = sqrt(a_bar_t)*x0 + sqrt(1-a_bar_t)*eps."""
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * eps, eps

    def loss(self, model, x0):
        """Sample random t, corrupt x0, predict noise, return MSE."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, eps = self.q_sample(x0, t)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_idx):
        """One reverse step: x_{t-1} from x_t."""
        t = torch.full((x_t.shape[0],), t_idx, device=self.device, dtype=torch.long)
        a = self.alphas[t_idx]
        a_bar = self.alpha_bars[t_idx]
        beta = self.betas[t_idx]

        eps_pred = model(x_t, t)
        mean = (1.0 / a.sqrt()) * (x_t - (beta / (1 - a_bar).sqrt()) * eps_pred)

        if t_idx == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + beta.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, n, device):
        """Full reverse chain starting from Gaussian noise."""
        model.eval()
        x = torch.randn(n, 3, 32, 32, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        model.train()
        return x
