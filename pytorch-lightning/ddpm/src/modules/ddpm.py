# import necessary packages
import torch
from torch import Tensor

class NoiseScheduler():
    def __init__(self, scheduler='linear') -> None:
        self.scheduler = scheduler
        self.betas = self._interpolate_beta()
        self.alphas = (1 - self.betas)
        self.alpha_bars = (torch.cumprod(self.alphas, axis=0))

    def _interpolate_beta(self) -> Tensor:
        match self.scheduler:
            case 'linear':
                return self._linear_beta_scheduler()
            case _:
                raise NotImplementedError(f"Scheduler {self.scheduler} not implemented")
            
    def _linear_beta_scheduler(self, timesteps=300, start=1e-4, end=2e-2) -> Tensor:
        return torch.linspace(start, end, timesteps)
    
    # TODO other scheduler functions
    
    def forward_diffusion(self, x_0: Tensor, t: int) -> Tensor:
        eps = torch.rand_like(x_0)
        # parametric trick: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps
        x_t = torch.sqrt(self.alpha_bars[t]) * x_0 + torch.sqrt(1 - self.alpha_bars[t]) * eps
        return x_t, eps