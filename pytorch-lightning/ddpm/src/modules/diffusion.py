import torch
from einops import rearrange

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_min_alpha_bar = torch.sqrt(1. - self.alpha_bar)

    def forward_process(self, x_0, noise, t):
        original_shape = x_0.shape #BxCxHxW
        batch_size = x_0.shape[0]
        
        assert t.shape[0] == batch_size
        sqrt_alpha_bar = rearrange(self.sqrt_alpha_bar.to(x_0.device)[t], 't -> t 1 1 1')
        sqrt_one_min_alpha_bar = rearrange(self.sqrt_one_min_alpha_bar.to(x_0.device)[t], 't -> t 1 1 1')
        
        return sqrt_alpha_bar * x_0 + sqrt_one_min_alpha_bar*noise
    
    def reverse_process(self, x_t, noise_pred, t):
        #print("debug", x_t.shape, noise_pred.shape, t.shape)
        x_0_bar = (x_t - self.sqrt_one_min_alpha_bar.to(x_t.device)[t]*noise_pred) / self.sqrt_alpha_bar.to(x_t.device)[t]
        x_0_bar = torch.clamp(x_0_bar, -1, 1)

        mu_theta = (x_t - (self.betas.to(x_t.device)[t]*noise_pred/self.sqrt_one_min_alpha_bar.to(x_t.device)[t]))/torch.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mu_theta, x_0_bar
        else:
            sigma = torch.sqrt(self.betas.to(x_t.device)[t]*(1 - self.alpha_bar.to(x_t.device)[t-1])/(1- self.alpha_bar.to(x_t.device)[t]))
            z = torch.randn_like(x_t).to(x_t.device)
            return mu_theta + sigma*z, x_0_bar