import torch


class FlattenedDistribution(torch.distributions.Distribution):
    def __init__(self, base_dist):
        super().__init__(batch_shape=base_dist.batch_shape, event_shape=base_dist.event_shape)
        self.base_dist = base_dist

    def log_prob(self, value):
        # Flatten all dimensions after the batch dimensions
        # We assume value is (Batch..., C, H, W) and base_dist expects (Batch..., D)
        # where D = C*H*W
        flat_value = value.reshape(value.shape[:-3] + (-1,))
        return self.base_dist.log_prob(flat_value)
    
    def anisotropy_loss(self):
        if hasattr(self.base_dist, "anisotropy_loss"):
            return self.base_dist.anisotropy_loss()
        else:
            raise NotImplementedError("Base distribution does not implement anisotropy_loss computation.")


class ScaledDistribution(torch.distributions.Distribution):
    """
    Density Rescaling Wrapper:
    p(z) = p_base((z - loc) / scale) / scale^D
    
    This avoids constructing covariance matrices with tiny eigenvalues by
    evaluating the density on the normalized scale.
    """
    def __init__(self, base_dist, loc, scale):
        super().__init__(batch_shape=base_dist.batch_shape, event_shape=base_dist.event_shape)
        self.base_dist = base_dist
        self.loc = loc
        self.scale = scale
        self.event_dim = self.event_shape.numel()

    def log_prob(self, value):
        # value: [Batch..., D] or [Batch..., C, H, W]
        # loc: broadcastable to value
        # scale: scalar or broadcastable
        
        z_norm = (value - self.loc) / self.scale
        
        # log p(z) = log p_base(z_norm) - D * log(scale)
        log_prob_base = self.base_dist.log_prob(z_norm)
        log_det_jacobian = self.event_dim * torch.log(self.scale)
        
        return log_prob_base - log_det_jacobian
    
    def anisotropy_loss(self):
        if hasattr(self.base_dist, "anisotropy_loss"):
            return self.base_dist.anisotropy_loss()
        else:
            raise NotImplementedError("Base distribution does not implement anisotropy_loss computation.")
