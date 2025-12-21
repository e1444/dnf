import torch


class FlattenedDistribution(torch.distributions.Distribution):
    arg_constraints = {}    # type: ignore
    
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
    arg_constraints = {}    # type: ignore
    
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


class MixtureDistribution(torch.distributions.Distribution):
    arg_constraints = {}    # type: ignore
    
    def __init__(self, component_dists, mixing_logits):
        """
        component_dists: list of torch.distributions.Distribution
        mixing_logits: Tensor of shape (B,K) or (K,) representing log mixing weights
        """
        super().__init__(batch_shape=component_dists[0].batch_shape, event_shape=component_dists[0].event_shape)
        self.component_dists = component_dists
        self.K = len(component_dists)
        self.mixing_logits = mixing_logits
        
    def log_prob(self, value):
        # value: [B, D] or [B, C, H, W]
        B = value.shape[0]
        
        log_probs = []
        for k in range(self.K):
            log_p_k = self.component_dists[k].log_prob(value)  # [B]
            log_probs.append(log_p_k.unsqueeze(1))  # [B, 1]
        
        log_probs = torch.cat(log_probs, dim=1)  # [B, K]
        
        if self.mixing_logits.dim() == 1:
            mixing_logits = self.mixing_logits.unsqueeze(0).expand(B, -1)  # [B, K]
        else:
            mixing_logits = self.mixing_logits  # [B, K]
        
        log_weights = mixing_logits - torch.logsumexp(mixing_logits, dim=1, keepdim=True)  # [B, K]
        weighted_log_probs = log_probs + log_weights  # [B, K]
        
        # log p(x) = logsumexp_k [ log p_k(x) + log w_k ]
        log_prob = torch.logsumexp(weighted_log_probs, dim=1)  # [B]
        
        return log_prob
    
    def anisotropy_loss(self):
        """
        Compute average anisotropy loss across mixture components, unweighted.
        """
        losses = []
        for k in range(self.K):
            if hasattr(self.component_dists[k], "anisotropy_loss"):
                losses.append(self.component_dists[k].anisotropy_loss().unsqueeze(0))
            else:
                raise NotImplementedError(f"Component distribution {k} does not implement anisotropy_loss computation.")
        return torch.cat(losses, dim=0).mean()
    
    @property
    def loc(self):
        weighted_loc = torch.stack([dist.loc for dist in self.component_dists], dim=0)  # [K, ...]
        if self.mixing_logits.dim() == 1:
            mixing_weights = torch.softmax(self.mixing_logits, dim=0)  # [K]
        else:
            mixing_weights = torch.softmax(self.mixing_logits.mean(dim=0), dim=0)  # [K]
        weighted_loc = (mixing_weights.view(-1, *([1] * (weighted_loc.dim() - 1))) * weighted_loc).sum(dim=0)
        return weighted_loc