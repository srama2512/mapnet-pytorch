import torch
import torch.nn as nn
import torch.nn.functional as F

class AggregateModule(nn.Module):
    def __init__(self, hidden_size, restrict_update=False, normalize_map_features=False):
        super().__init__()
        self.restrict_update = restrict_update
        self.normalize_map_features = normalize_map_features

    def forward(self, x, hidden):
        """
        Inputs:
            x      - (bs, hidden_size)
            hidden - (bs, hidden_size)
        """
        x1 = self._compute_forward(x, hidden) # (bs, hidden_size)
        # Retain hidden state wherever no updates are needed
        if self.restrict_update:
            x_mask = (x != 0).float()
            x1 =  x_mask * x1 + (1-x_mask) * hidden
        # Normalize map features if necessary
        if self.normalize_map_features:
            x1 = F.normalize(x1 + 1e-10, dim=1)
        return x1

    def _compute_forward(self, x, hidden):
        raise NotImplementedError

class GRUAggregate(AggregateModule):
    def __init__(
        self,
        hidden_size,
        restrict_update=False,
        normalize_map_features=False
    ):
        super().__init__(
            hidden_size,
            restrict_update=restrict_update,
            normalize_map_features=normalize_map_features
        )
        self.main = nn.GRU(hidden_size, hidden_size, num_layers=1)

    def _compute_forward(self, x, hidden):
        x1, _ = self.main(x.unsqueeze(0), hidden.unsqueeze(0))
        return x1[0]
