import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical


class ConcreteSelector(nn.Module):
    '''
    Input layer that selects features by learning a binary matrix, based on [1].

    [1] Concrete Autoencoders for Differentiable Feature Selection and
    Reconstruction (Balin et al., 2019)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      lr_multiplier: to increase learning rate for Concrete parameters.
    '''
    def __init__(self, input_size, k, temperature=10.0, lr_multiplier=3.0):
        super().__init__()
        self._logits = nn.Parameter(torch.zeros(k, input_size, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature
        self.lr_multiplier = lr_multiplier
        
    @property
    def logits(self):
        return self._logits * self.lr_multiplier

    # TODO consider making temperature an argument to forward
    def forward(self, x):
        # Sample selection matrix.
        M = self.sample(len(x))

        # Apply selection matrix.
        return (x.unsqueeze(1) @ M.permute(0, 2, 1)).squeeze(1)

    def sample(self, n_samples):
        '''Sample approximate binary matrices.'''
        dist = RelaxedOneHotCategorical(self.temperature, logits=self.logits)
        return dist.rsample(torch.Size([n_samples]))

    def get_inds(self):
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)
        
        
class ConcreteMask(nn.Module):
    '''
    Input layer that selects features by learning a binary mask, based on [2].

    [2] Predictive and Robust Gene Selection for Spatial Transcriptomics
    (Covert et al., 2023)

    Args:
      input_size: number of inputs.
      k: number of features to be selected.
      temperature: temperature for Concrete samples.
      lr_multiplier: to increase learning rate for Concrete parameters.
    '''
    def __init__(self, input_size, k, temperature=10.0, lr_multiplier=3.0):
        super().__init__()
        self._logits = nn.Parameter(torch.zeros(k, input_size, requires_grad=True))
        self.input_size = input_size
        self.k = k
        self.output_size = k
        self.temperature = temperature
        self.lr_multiplier = lr_multiplier
        
    @property
    def logits(self):
        return self._logits * self.lr_multiplier

    # TODO consider making temperature an argument to forward
    def forward(self, x):
        # Sample selection mask.
        M = self.sample(len(x))

        # Apply selection mask.
        return x * M

    def sample(self, n_samples):
        '''Sample approximate binary masks.'''
        dist = RelaxedOneHotCategorical(self.temperature, logits=self.logits)
        sample = dist.rsample(torch.Size([n_samples]))
        return torch.max(sample, axis=1).values

    def get_inds(self):
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return 'input_size={}, temperature={}, k={}'.format(
            self.input_size, self.temperature, self.k)
