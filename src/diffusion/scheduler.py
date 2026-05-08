import torch


class DiffusionScheduler:
    def __init__(self, T=50, beta_start=1e-4, beta_end=0.02):
        self.T = T

        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self
