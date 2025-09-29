import torch
import torch.nn.functional as F

class FM:
    
    def __init__(self, sigma_min=1e-5, timescale=1.0):
        self.sigma_min = sigma_min
        self.prediction_type = None
        self.timescale = timescale
    
    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)
    
    def get_betas(self, n_timesteps):
        return torch.zeros(n_timesteps) # Not VP and not supported
    
    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        target = self.A(t) * x + self.B(t) * noise # -dxt/dt
        if return_loss_unreduced:
            loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - target.float()) ** 2).mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss

    def mask_loss(self, net, x, mask, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        x_t = x * (1 - mask) + x_t * mask
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        pred = x * (1 - mask) + pred * mask
        
        target = self.A(t) * x + self.B(t) * noise # -dxt/dt
        target = x * (1 - mask) + target * mask
        if return_loss_unreduced:
            loss = ((pred.float() - target.float()) ** 2)
            loss = loss[mask].mean(dim=[1, 2])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - target.float()) ** 2)
            loss = loss[mask].mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def mask_mae_loss(self, net, x, target, mask, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        dif_target = self.A(t) * target + self.B(t) * noise # -dxt/dt
        if return_loss_unreduced:
            loss = ((pred.float() - dif_target.float()) ** 2)[~mask].mean(dim=[1, 2]) + 0.1 * ((pred.float() - self.add_noise(target, t)[0].float()) ** 2)[mask].mean(dim=[1, 2])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - dif_target.float()) ** 2)[~mask] + 0.1 * ((pred.float() - self.add_noise(target, t)[0].float()) ** 2)[mask].mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def causal_loss(self, net, x, targets, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        target = self.A(t) * targets + self.B(t) * noise # -dxt/dt
        if return_loss_unreduced:
            loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - target.float()) ** 2).mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def get_prediction(
        self,
        net,
        x_t,
        t,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
    ):
        if net_kwargs is None:
            net_kwargs = {}
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        if guidance != 1.0:
            assert uncond_net_kwargs is not None
            uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
            pred = uncond_pred + guidance * (pred - uncond_pred)
        return pred
    
    def convert_sample_prediction(self, x_t, t, pred):
        M = torch.tensor([
            [self.alpha(t), self.sigma(t)],
            [self.A(t), self.B(t)],
        ], dtype=torch.float64)
        M_inv = torch.linalg.inv(M)
        sample_pred = M_inv[0, 0].item() * x_t + M_inv[0, 1].item() * pred
        return sample_pred

class FMEulerSampler:

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(
        self,
        net,
        shape,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
    ):
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                t = t_steps[i].repeat(x_t.shape[0])
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
        return x_t
    
    def inpaint(
        self,
        net,
        z,
        mask,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
        clean=False,
    ):
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        device = next(net.parameters()).device
        x_t = torch.randn(z.shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                t = t_steps[i].repeat(x_t.shape[0])
                if clean:
                    x_t = z * (1 - mask) + x_t * mask
                else:
                    z_t, _ = self.diffusion.add_noise(z, t)
                    x_t = z_t * (1 - mask) + x_t * mask
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
        
        # final add back in clean context
        x_t = z * (1 - mask) + x_t * mask
        return x_t
    
    def masked_sample(
        self,
        net,
        shape,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
    ):
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                if i == 0:
                    cur_net_kwargs = net_kwargs | {'mask': torch.ones(shape).long()}
                    cur_uncond_net_kwargs = uncond_net_kwargs | {'mask': torch.ones(shape).long()}
                else:
                    cur_net_kwargs = net_kwargs
                    cur_uncond_net_kwargs = uncond_net_kwargs
                t = t_steps[i].repeat(x_t.shape[0])
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=cur_net_kwargs,
                    uncond_net_kwargs=cur_uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
        return x_t
    
    def masked_inpaint(
        self,
        net,
        z,
        mask,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
    ):
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        device = next(net.parameters()).device
        x_t = torch.randn(z.shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

        with torch.no_grad():
            for i in range(n_steps):
                if i == 0:
                    cur_net_kwargs = net_kwargs | {'mask': mask.long()}
                    cur_uncond_net_kwargs = uncond_net_kwargs | {'mask': mask.long()}
                else:
                    cur_net_kwargs = net_kwargs
                    cur_uncond_net_kwargs = uncond_net_kwargs
                t = t_steps[i].repeat(x_t.shape[0])
                z_t, _ = self.diffusion.add_noise(z, t)
                x_t = z_t * (1 - mask) + x_t * mask
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=cur_net_kwargs,
                    uncond_net_kwargs=cur_uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
        return x_t