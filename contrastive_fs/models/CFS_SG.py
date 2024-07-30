import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from itertools import chain
import math
import torch
import numpy as np

class CFS_SG(pl.LightningModule):
    '''MLP with input layer selection.

    Args:
      input_layer: input layer type (e.g., 'concrete_gates').
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between hidden layers.
      output_activation: nonlinearity at output.
      kwargs: additional arguments (e.g., k, init, append). Some are optional,
        but k is required for ConcreteMask and ConcreteGates.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 lam,
                 k_prime,
                 lr,
                 loss_fn):
        super().__init__()

        # Set all gates to have mean of 0.5 initially
        self.mu = nn.Parameter(
            torch.normal(mean=0.5, std=0.01, size=(input_size,)),
            requires_grad=True
        )
        self.sigma = 0.5 # Default from STG repo
        self.lam = lam
        self.lr = lr

        self.background_input_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, k_prime),
        )

        fc_layers = []
        for d_in, d_out in zip([input_size+k_prime] + hidden, hidden + [output_size]):
            fc_layers.append(nn.Linear(d_in, d_out))
            fc_layers.append(nn.ReLU())
        fc_layers = fc_layers[:-1]


        self.fc = nn.Sequential(*fc_layers)
        self.loss_fn = loss_fn
        self.avg_background_gate_vals = 0
        self.avg_salient_gate_vals = 0
        self.num_selected_gates = 0

    def forward(self, x, **kwargs):
        test = self.input_layer(x)
        return self.fc(test)

    def on_train_batch_start(self, batch, batch_idx: int):
        if self.avg_salient_gate_vals > 0.99:
            return -1

    def reg(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_tar = x[y != 0]
        x_bg = x[y == 0]

        base_noise = torch.normal(mean=0.0, std=1.0, size=self.mu.shape).to(self.device)
        gate_vals = self.mu + self.sigma * base_noise

        # Applying the "hard sigmoid" to the initial gate values
        gate_vals = torch.clip(gate_vals, min=0, max=1)

        masked_x_tar = x_tar * gate_vals
        target_embedding = self.background_input_layer(x_tar).detach()
        background_embedding = self.background_input_layer(x_bg)

        target_recon = self.fc(
            torch.cat([target_embedding, masked_x_tar], dim=1)
        )
        background_recon = self.fc(
            torch.cat(
                [
                    background_embedding,
                    torch.zeros(x_bg.shape[0], x_bg.shape[1]).to(self.device),
                ],
                dim=1,
            )
        )

        target_loss, background_loss = self.loss_fn(x_tar, target_recon), self.loss_fn(
            x_bg, background_recon
        )

        reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
        loss = target_loss + background_loss + self.lam * reg
        num_selected_gates = (self.mu > 0.9).sum()
        self.num_selected_gates = num_selected_gates
        self.log('num_selected_gates', num_selected_gates, prog_bar=True)
        return loss

    def get_inds(self, num_features):
        idx = np.argsort(self.mu.detach().numpy())[-num_features:]
        return idx

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
