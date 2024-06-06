# MIT License

# Copyright (c) 2022 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 10/18/2022
# RBFLayer implemented in JAX/FLAX

from typing import Callable
import jax
import jax.numpy as jnp
from flax import linen as nn


# RBF kernel functions
@jax.jit
def gaussian(alpha):
    phi = jnp.exp(-1 * alpha**2)
    return phi

@jax.jit
def gaussian_wide(alpha):
    phi = jnp.exp(-0.1 * alpha**2)
    return phi

@jax.jit
def gaussian_wider(alpha):
    phi = jnp.exp(-0.01 * alpha**2)
    return phi

@jax.jit
def inverse_quadratic(alpha):
    phi = jnp.ones_like(alpha) / (jnp.ones_like(alpha) + alpha**2)
    return phi

@jax.jit
def linear(alpha):
    phi = alpha
    return phi


@jax.jit
def quadratic(alpha):
    phi = alpha**2
    return phi


@jax.jit
def multiquadric(alpha):
    phi = (jnp.ones_like(alpha) + alpha**2)**0.5
    return phi


@jax.jit
def inverse_multiquadric(alpha):
    phi = jnp.ones_like(alpha) / (jnp.ones_like(alpha) + alpha**2)**0.5
    return phi


@jax.jit
def spline(alpha):
    phi = alpha**2 * jnp.log(alpha + jnp.ones_like(alpha))
    return phi


@jax.jit
def poisson_one(alpha):
    phi = (alpha - jnp.ones_like(alpha)) * jnp.exp(-alpha)
    return phi


@jax.jit
def poisson_two(alpha):
    phi = (
        ((alpha - 2 * jnp.ones_like(alpha)) / 2 * jnp.ones_like(alpha))
        * alpha
        * jnp.exp(-alpha)
    )
    return phi


@jax.jit
def matern32(alpha):
    phi = (jnp.ones_like(alpha) + 3**0.5 * alpha) * jnp.exp(-(3**0.5) * alpha)
    return phi


@jax.jit
def matern52(alpha):
    phi = (jnp.ones_like(alpha) + 5**0.5 * alpha + (5 / 3) * alpha**2) * jnp.exp(
        -(5**0.5) * alpha
    )
    return phi


class RBFNet(nn.Module):
    """
    RBF Network, defined by a layer of RBF kernels and a linear layer

    Args:
        in_features (int): number of features in the input vector
        out_features (int): number of features in the output vector
        num_kernels (int): number of kernels in RBF layer
        basis_func (Callable): radial basis function to use
    """

    in_features: int
    out_features: int
    num_kernels: int
    basis_func: Callable

    def setup(self):
        self.centers = self.param(
            "centers",
            nn.initializers.normal(1.0),
            (self.num_kernels, self.in_features),
        )
        self.log_sigs = self.param(
            "log_sigs",
            nn.initializers.constant(0.0),
            (self.num_kernels,),
        )
        self.linear = nn.Dense(self.out_features)

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """
        batch_size = x.shape[0]

        # expand centers to (batch_size, num_kernels, in_features)
        c = jnp.broadcast_to(
            self.centers, (batch_size, self.num_kernels, self.in_features)
        )

        # expand input to (batch_size, num_kernels, in_features)
        x_e = jnp.broadcast_to(
            jnp.expand_dims(x, axis=1), (batch_size, self.num_kernels, self.in_features)
        )

        # distances to center
        d = ((((x_e - c) ** 2).sum(-1)) ** 0.5) / jnp.broadcast_to(
            jnp.exp(self.log_sigs), (batch_size, self.num_kernels)
        )

        # output of rbfs
        rbf_out = self.basis_func(d)
        
        # output of network
        out = self.linear(rbf_out)

        return out


class RBFLayerBatched(nn.Module):
    """
    RBF Layer, defined by a layer of RBF kernels

    Args:
        in_features (int): number of features in the input vector
        num_kernels (int): number of kernels in RBF layer
        basis_func (Callable): radial basis function to use
    """

    in_features: int
    num_kernels: int
    basis_func: Callable

    def setup(self):
        self.centers = self.param(
            "centers",
            nn.initializers.normal(1.0),
            (self.num_kernels, self.in_features),
        )
        self.log_sigs = self.param(
            "log_sigs",
            nn.initializers.constant(0.0),
            (self.num_kernels,),
        )

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """
        batch_size = x.shape[0]

        # expand centers to (batch_size, num_kernels, in_features)
        c = jnp.broadcast_to(
            self.centers, (batch_size, self.num_kernels, self.in_features)
        )

        # expand input to (batch_size, num_kernels, in_features)
        x_e = jnp.broadcast_to(
            jnp.expand_dims(x, axis=1), (batch_size, self.num_kernels, self.in_features)
        )

        # distances to center
        d = ((((x_e - c) ** 2).sum(-1)) ** 0.5) / jnp.broadcast_to(
            jnp.exp(self.log_sigs), (batch_size, self.num_kernels)
        )

        # output of rbfs
        rbf_out = self.basis_func(d)

        return rbf_out


class RBFLayer(nn.Module):
    """
    RBF Layer, defined by a layer of RBF kernels

    Args:
        in_features (int): number of features in the input vector
        num_kernels (int): number of kernels in RBF layer
        basis_func (Callable): radial basis function to use
    """

    in_features: int
    num_kernels: int
    basis_func: Callable

    def setup(self):
        self.centers = self.param(
            "centers",
            nn.initializers.normal(1.0),
            (self.num_kernels, self.in_features),
        )
        self.log_sigs = self.param(
            "log_sigs",
            nn.initializers.constant(0.0),
            (self.num_kernels,),
        )

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (in_features, ))

        Returns:
            rbf_out (out vector, jnp.DeviceArray (num_kernels, ))
        """

        # expand centers to (batch_size, num_kernels, in_features)
        # c = jnp.broadcast_to(
        #     self.centers, (self.num_kernels, self.in_features)
        # )

        # expand input to (batch_size, num_kernels, in_features)
        x_e = jnp.broadcast_to(
            x, (self.num_kernels, self.in_features)
        )

        # distances to center
        d = ((((x_e - self.centers) ** 2).sum(-1)) ** 0.5) / jnp.exp(self.log_sigs)

        # output of rbfs
        rbf_out = self.basis_func(d)

        return rbf_out