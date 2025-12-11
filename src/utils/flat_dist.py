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