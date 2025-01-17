"""
Created by Hamid Eghbal-zadeh at 05.02.21
Johannes Kepler University of Linz
"""

import torch
import torch.nn as nn

import numpy as np

from carla.agents.utils.weight_init import weights_init_he, weights_init_xavier, weights_init_kaiming_normal
from carla.architectures.sebastian.InvRes import Encoder, InvertedResidualVAEModel
from carla.architectures.vae import BasicBetaVAE


def mlp(sizes, activation, output_activation=nn.Identity, normalize_input=False):
    # remove zero in sizes
    sizes = [e for e in sizes if e > 0]
    layers = []
    if normalize_input:
        layers += [nn.BatchNorm1d(sizes[0])]
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def cnn(sizes, obs_shape, out_hidden, activation, kernel_size, stride, output_activation=nn.Identity, bottleneck=False):
    # remove zero in sizes
    sizes = [e for e in sizes if e > 0]
    layers = []
    for j in range(len(sizes) - 1):
        layers += [nn.Conv2d(sizes[j], sizes[j + 1], kernel_size=kernel_size, stride=stride, padding=1), activation()]

    if bottleneck:
        # infer shape
        feat_final_dim = np.prod(nn.Sequential(*layers)(torch.zeros(1, *obs_shape).permute(0, 3, 1, 2)).shape)
        layers += [nn.Flatten(), nn.Linear(feat_final_dim, out_hidden), output_activation()]

    return nn.Sequential(*layers)

# class Permute(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.dims = dims,  # extra comma

#     def forward(self, x):
#         return x.permute(*self.dims)


def invres(obs_shape, out_hidden, output_activation=nn.Identity):

    layers = [  # Permute([0,3,1,2]),
        Encoder(out_hidden, obs_shape[-1], obs_shape[0]), output_activation()
    ]
    return nn.Sequential(*layers)


def invres_nobn(obs_shape, out_hidden, output_activation=nn.Identity):

    layers = [  # Permute([0,3,1,2]),
        Encoder(out_hidden, obs_shape[-1], obs_shape[0], norm_layer=nn.Identity), output_activation()
    ]
    return nn.Sequential(*layers)


def get_clf_model(init, latent_dim, type, n_context=90):
    from carla.architectures.clf import LinearContextClf, MLPContextClf
    if type == 'linear':
        model = LinearContextClf(n_latent=latent_dim, n_context=n_context)
    elif type == 'mlp':
        model = MLPContextClf(n_latent=latent_dim, n_context=n_context)
    else:
        raise NotImplementedError('{} is not implemented!'.format(type))

    if init == 'he':
        model.init_weights(weights_init_he)
        model.apply(weights_init_he)
    elif init == 'xavier':
        model.init_weights(weights_init_xavier)
        model.apply(weights_init_xavier)
    elif init == 'kaiming':
        model.init_weights(weights_init_kaiming_normal)
        model.apply(weights_init_kaiming_normal)
    return model


def get_model(kernel_sizes, output_padding_lst, latent_dim, hidden_dims, img_size, loss_type, vae_gamma, vae_beta,
              nonlin, enc_bn, dec_bn, dec_out_nonlin, prior, soft_clip, init, strides, paddings, batch_size,
              vae_c_max=25, vae_c_stop_iter=100, vae_geco_goal=0.5, vae_reduction='mean', **kwargs):
    model = BasicBetaVAE(in_channels=3, kernel_sizes=kernel_sizes, latent_dim=latent_dim, hidden_dims=hidden_dims,
                         loss_type=loss_type, img_size=img_size, nonlin=nonlin, enc_bn=enc_bn, dec_bn=dec_bn,
                         output_padding_lst=output_padding_lst, gamma=vae_gamma, beta=vae_beta,
                         dec_out_nonlin=dec_out_nonlin, prior=prior, soft_clip=soft_clip, strides=strides,
                         paddings=paddings, batch_size=batch_size, max_capacity=vae_c_max,
                         vae_c_stop_iter=vae_c_stop_iter, geco_goal=vae_geco_goal, reduction=vae_reduction)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

    d = {}
    for module in model.named_modules():
        n = module[1].__class__.__name__
        print(n, count_parameters(module[1]))
        if not(n in d):
            d[n] = 0

        d[n] += 1

    print(d)
    raise "nope"

    if init == 'he':
        model.init_weights(weights_init_he)
        model.apply(weights_init_he)
    elif init == 'xavier':
        model.init_weights(weights_init_xavier)
        model.apply(weights_init_xavier)
    elif init == 'kaiming':
        model.init_weights(weights_init_kaiming_normal)
        model.apply(weights_init_kaiming_normal)
    return model


# def get_model(latent_dim, img_size, **kwargs):

#     model = InvertedResidualVAEModel(num_latents=latent_dim, num_channels=3, image_size=img_size)

#     # init not important if we use pretrained
#     # if init == 'he':
#     #     model.init_weights(weights_init_he)
#     #     model.apply(weights_init_he)
#     # elif init == 'xavier':
#     #     model.init_weights(weights_init_xavier)
#     #     model.apply(weights_init_xavier)
#     # elif init == 'kaiming':
#     #     model.init_weights(weights_init_kaiming_normal)
#     #     model.apply(weights_init_kaiming_normal)
#     return model
