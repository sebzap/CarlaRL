import torch
from torch import nn
import torch.nn.init as init


def _init_layer(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim, num_channels, image_size):
        super().__init__()

        self._latent_dim = latent_dim
        self._num_channels = num_channels
        self._image_size = image_size

    def forward(self, *input):
        raise NotImplementedError

    def latent_dim(self):
        return self._latent_dim

    def num_channels(self):
        return self._num_channels

    def image_size(self):
        return self._image_size

    def init_layers(self):
        for block in self._modules:
            from collections.abc import Iterable
            if isinstance(self._modules[block], Iterable):
                for m in self._modules[block]:
                    _init_layer(m)
            else:
                _init_layer(self._modules[block])


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        return x


class Unsqueeze3D(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAEModule(nn.Module):
    """
    pytorch Module for Variational Auto Encoders
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, activation=nn.Sigmoid(), loss_activation: nn.Module = None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.activation = activation
        self.use_model_regulariser = False
        self.loss_activation = loss_activation if loss_activation is not None else self.activation

    def encode(self, x: torch.Tensor):
        """Encodeds Input to latents

        Args:
            x (Tensor): input

        Returns:
            Tensor: latent encoding
        """
        return self.encoder(x)

    def decode(self, z):
        """Decondes latent z into output (with (sigmoid) activation)

        Args:
            z (Tensor): latent

        Returns:
            Tensor: output
        """
        return self.activation(self.decoder(z))

    def reparametrize(self, mu, logvar):
        """Reparamterization trick for training

        Args:
            mu (Tensor): mean of latents
            logvar (Tensor): log variance of latents

        Returns:
            Tensor: latent sample
        """
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass from input to reconstruction (including reparamterization trick).
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var), z
        """
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        # NOTE: we use decodeR(!) here not decode (which includes sigmoid activation)
        return self.loss_activation(self.decoder(z)), (mu, logvar), z

    def reconstruct(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var)
        """
        mu, logvar = self.encode(x)
        return self.activation(self.decoder(mu)), (mu, logvar)

    def eval_forward(self, x):
        """Forward pass from input to reconstruction, without reparamterization trick.
        Only uses mu, ignores logvar
        Omits (sigmoid) activation for the use of BCEWithLogitsLoss

        Args:
            x (Tensor): input

        Returns:
            Tensor, (Tensor, Tensor): reconstruction, (mu, log_var)
        """
        mu, logvar = self.encode(x)
        return self.loss_activation(self.decoder(mu)), (mu, logvar)

    def model_regulariser(self):
        return 0
