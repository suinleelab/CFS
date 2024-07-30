import pytorch_lightning as pl
from contrastive_fs.layers import ConcreteSelector
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import math
import torch
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_size, k):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, k),
        )

        self.decoder = nn.Sequential(
            nn.Linear(k, 100),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        recon = self.decoder(embedding)

        return embedding, recon


class CFS_Pretrained(pl.LightningModule):
    def __init__(self,
                 input_size,
                 output_size,
                 max_pretrain_epochs,
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
        self.lr = lr
        self.max_pretrain_epochs = max_pretrain_epochs
        print(self.max_pretrain_epochs)

        self.background_autoencoder = Autoencoder(input_size=input_size, k=k_prime)

        fc_layers = []
        for d_in, d_out in zip([input_size + k_prime] + hidden, hidden + [output_size]):
            fc_layers.append(nn.Linear(d_in, d_out))
            fc_layers.append(nn.ReLU())
        fc_layers = fc_layers[:-1]


        self.fc = nn.Sequential(*fc_layers)
        self.loss_fn = loss_fn
        self.lam = lam
        self.num_selected_gates = 0

    def forward(self, x, **kwargs):
        base_noise = torch.normal(mean=0.0, std=1.0, size=self.mu.shape).to(self.device)
        gate_vals = self.mu + self.sigma * base_noise

        # Applying the "hard sigmoid" to the initial gate values
        gate_vals = torch.clip(gate_vals, min=0, max=1)

        gated_x = x * gate_vals

        bg_representation, _ = self.background_autoencoder(x)
        bg_representation = bg_representation.detach()
        return self.fc(torch.cat([gated_x, bg_representation], dim=1))

    def reg(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.current_epoch < self.max_pretrain_epochs:
            # When pretraining, only use the background samples
            import pdb
            pdb.set_trace()
            x = x[y == 0]
            if len(x) == 0:
                return 0
            else:
                _, recon = self.background_autoencoder(x)
                loss = F.mse_loss(x, recon)
                self.log('bg_recon_loss', loss)
                return loss
        else:
            """
            # After pretraining, only use the target samples
            x = x[y != 0]
            output = self(x)
            self.input_layer.temperature *= self.r

            loss = self.loss_fn(output, x)
            self.log('Loss', loss, prog_bar=True)

            M = self.input_layer.sample(n_samples=256)
            values = torch.mean(M, dim=0)
            self.avg_gate_vals = torch.max(values, dim=1).values.mean()

            self.log('AvgGateVals', self.avg_gate_vals, prog_bar=True)
            """
            x_tar = x[y != 0]
            x_bg = x[y == 0]

            base_noise = torch.normal(mean=0.0, std=1.0, size=self.mu.shape).to(self.device)
            gate_vals = self.mu + self.sigma * base_noise

            # Applying the "hard sigmoid" to the initial gate values
            gate_vals = torch.clip(gate_vals, min=0, max=1)

            masked_x_tar = x_tar * gate_vals

            x_tar_recon = self.fc(
                torch.cat([masked_x_tar, self.background_autoencoder(x_tar)[0].detach()], dim=1))

            if len(x_bg > 0):
                x_bg_recon = self.fc(torch.cat([
                    torch.zeros(x_bg.shape[0], x_bg.shape[1]).to(self.device),
                    self.background_autoencoder(x_bg)[0].detach()
                ], dim=1))

                loss_bg = self.loss_fn(x_bg, x_bg_recon)
            else:
                loss_bg = 0

            if self.max_pretrain_epochs == 0:
                loss_bg = 0
            loss_tar = self.loss_fn(x_tar, x_tar_recon)

            reg = torch.mean(self.reg((self.mu + 0.5) / self.sigma))
            loss = loss_tar + loss_bg + self.lam * reg

            num_selected_gates = (self.mu > 0.9).sum()
            self.num_selected_gates = num_selected_gates
            self.log('num_selected_gates', num_selected_gates, prog_bar=True)
            return loss

    def get_inds(self, num_features):
        idx = np.argsort(self.mu.detach().numpy())[-num_features:]
        return idx

    def on_train_batch_start(self, batch, batch_idx: int):
        if self.num_selected_gates >= self.k:
            return -1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
