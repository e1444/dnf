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

    def kl_to_isotropic(self, tau=1.0):
        if not hasattr(self.base_dist, 'kl_to_isotropic'):
            raise NotImplementedError("Base distribution does not implement kl_to_isotropic method.")
        
        return self.base_dist.kl_to_isotropic(tau=tau)
    
    def anisotropy_penalty(self):
        if not hasattr(self.base_dist, 'anisotropy_penalty'):
            raise NotImplementedError("Base distribution does not implement anisotropy_penalty method.")
        
        return self.base_dist.anisotropy_penalty()