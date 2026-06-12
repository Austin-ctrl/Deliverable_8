import torch
from config import LR


class SharedRMSProp(torch.optim.RMSprop):
    """
    RMSProp with running statistics shared across all worker processes.
    Recommended over per-thread statistics in the A3C paper (Section 4).
    """

    def __init__(self, params, lr=LR, alpha=0.99, eps=1e-5):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps)
        # Move running stats into shared memory so all workers read/write them
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()