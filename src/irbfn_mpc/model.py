# MIT License

# Copyright (c) 2023 Hongrui Zheng

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
# Last Modified: 10/27/2022
# WC-RBFN in FLAX

from functools import partial
from typing import Callable, Sequence, Optional

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax_rbf.flax_rbf import (
    RBFLayer,
    WarmstartCenteredRBFLayer,
    FixedCenteredRBFLayer,
    FixedWidthCenteredRBFLayer,
)


@partial(jax.jit, static_argnums=[1, 2, 3, 4, 5, 6])
@chex.assert_max_traces(n=2)
def _region_activation(
    x,
    num_regions,
    num_split_dimensions,
    lower_bounds,
    upper_bounds,
    delta,
    dimension_ranges,
):
    """
    Smooth indicator function gamma

    Args:
        x (jnp.DeviceArray (batch_size, in_features)): input vector
        num_regions,
        num_split_dimensions,
        lower_bounds,
        upper_bounds,
        delta,
        dimension_ranges,

    Returns:
        region_weights (jnp.DeviceArray (batch_size, num_regions)): region weights for each input
    """

    # final gamma
    out_gamma = jnp.zeros((x.shape[0], num_regions))

    # gammas for each dimension
    all_gammas = []
    for d in range(num_split_dimensions):
        lower_diffs = jnp.broadcast_to(
            x[:, d], (len(lower_bounds[d]), x.shape[0])
        ).T - jnp.array(lower_bounds[d])
        upper_diffs = (
            jnp.array(upper_bounds[d])
            - jnp.broadcast_to(x[:, d], (len(upper_bounds[d]), x.shape[0])).T
        )

        gamma = ((jnp.tanh(delta[d] * lower_diffs) + 1) / 2) * (
            ((jnp.tanh(delta[d] * upper_diffs) + 1) / 2)
        )
        all_gammas.append(gamma)

    for i in range(len(dimension_ranges)):
        curr_gamma = all_gammas[0][:, dimension_ranges[i][0]]
        for j in range(1, num_split_dimensions):
            curr_gamma = curr_gamma * all_gammas[j][:, dimension_ranges[i][j]]

        out_gamma = out_gamma.at[:, i].set(curr_gamma)

    return out_gamma


class WCRBFNet(nn.Module):
    """
    RBF Network, defined by a layer of RBF kernels and a linear layer

    Args:
        in_features (int): number of features in the input vector
        out_features (int): number of features in the output vector
        num_kernels (int): number of kernels in each RBF layer
        basis_func (Callable): radial basis function to use
        num_regions (int): number of regions used
        region_bounds (Sequence[Sequence[float]]): upper and lower bounds of the regions
        split_indices: (Sequence[int]): indices of the input vector that is split into regions
        delta (float): scaling factor for region indicator function
    """

    in_features: int
    out_features: int
    num_kernels: int
    basis_func: Callable
    num_regions: int
    lower_bounds: Sequence[Sequence[float]]
    upper_bounds: Sequence[Sequence[float]]
    dimension_ranges: Sequence[Sequence[float]]
    activation_idx: Sequence[int]
    delta: Sequence[float]
    centers: Optional[jax.Array] = None
    fixed_centers: Optional[bool] = False
    fixed_width: Optional[bool] = False

    def setup(self):
        self.num_split_dimensions = len(self.activation_idx)

        # check type of layers
        if self.centers is not None:
            if self.fixed_centers:
                if self.fixed_width:
                    layer_fn = FixedWidthCenteredRBFLayer
                else:
                    layer_fn = FixedCenteredRBFLayer
            else:
                layer_fn = WarmstartCenteredRBFLayer
        else:
            layer_fn = RBFLayer

        # instantiate multi-headed RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            layer_fn,
            in_axes=None,
            out_axes=0,
            axis_size=self.num_regions,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        # Batching RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            broadcasted_rbf,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.rbf_list = broadcasted_rbf(
            in_features=self.in_features,
            num_kernels=self.num_kernels,
            basis_func=self.basis_func,
            centers_values=self.centers,
        )
        self.linear = nn.Dense(self.out_features)

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """

        # indicators
        gamma = _region_activation(
            x,
            self.num_regions,
            self.num_split_dimensions,
            self.lower_bounds,
            self.upper_bounds,
            self.delta,
            self.dimension_ranges,
        )
        gamma_rep = jnp.repeat(jnp.expand_dims(gamma, -1), self.num_kernels, axis=-1)

        # rbf networks
        all_x = self.rbf_list(x)

        # interpolation
        rbf_out = jnp.sum(gamma_rep * all_x, axis=1)

        # linear layer
        out = self.linear(rbf_out)

        return out


class DeeperWCRBFNet(nn.Module):
    """
    RBF Network, defined by a layer of RBF kernels and a linear layer

    Args:
        in_features (int): number of features in the input vector
        out_features (int): number of features in the output vector
        num_kernels (int): number of kernels in each RBF layer
        basis_func (Callable): radial basis function to use
        num_regions (int): number of regions used
        region_bounds (Sequence[Sequence[float]]): upper and lower bounds of the regions
        split_indices: (Sequence[int]): indices of the input vector that is split into regions
        delta (float): scaling factor for region indicator function
    """

    in_features: int
    out_features: int
    num_kernels: int
    basis_func: Callable
    num_regions: int
    lower_bounds: Sequence[Sequence[float]]
    upper_bounds: Sequence[Sequence[float]]
    dimension_ranges: Sequence[Sequence[float]]
    activation_idx: Sequence[int]
    delta: Sequence[float]

    def setup(self):
        self.num_split_dimensions = len(self.activation_idx)

        # instantiate multi-headed RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            RBFLayer,
            in_axes=None,
            out_axes=0,
            axis_size=self.num_regions,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        # Batching RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            broadcasted_rbf,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.rbf_list = broadcasted_rbf(
            in_features=self.in_features,
            num_kernels=self.num_kernels,
            basis_func=self.basis_func,
        )
        self.linear_pre1 = nn.Dense(64)
        self.linear_pre2 = nn.Dense(64)
        self.linear = nn.Dense(self.out_features)

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """

        # indicators
        gamma = _region_activation(
            x,
            self.num_regions,
            self.num_split_dimensions,
            self.lower_bounds,
            self.upper_bounds,
            self.delta,
            self.dimension_ranges,
        )
        gamma_rep = jnp.repeat(jnp.expand_dims(gamma, -1), self.num_kernels, axis=-1)

        # rbf networks
        all_x = self.rbf_list(x)

        # interpolation
        rbf_out = jnp.sum(gamma_rep * all_x, axis=1)

        # linear layers
        out_pre1 = self.linear_pre1(rbf_out)
        out_pre2 = self.linear_pre2(nn.relu(out_pre1))
        out = self.linear(nn.relu(out_pre2))

        return out


class MLP(nn.Module):
    """
    RBF Network, defined by a layer of RBF kernels and a linear layer

    Args:
        in_features (int): number of features in the input vector
        out_features (int): number of features in the output vector
        num_kernels (int): number of kernels in each RBF layer
        basis_func (Callable): radial basis function to use
        num_regions (int): number of regions used
        region_bounds (Sequence[Sequence[float]]): upper and lower bounds of the regions
        split_indices: (Sequence[int]): indices of the input vector that is split into regions
        delta (float): scaling factor for region indicator function
    """

    in_features: int
    out_features: int
    num_kernels: int
    basis_func: Callable
    num_regions: int
    lower_bounds: Sequence[Sequence[float]]
    upper_bounds: Sequence[Sequence[float]]
    dimension_ranges: Sequence[Sequence[float]]
    activation_idx: Sequence[int]
    delta: Sequence[float]

    def setup(self):
        self.linear1 = nn.Dense(int(self.num_kernels / 2))
        self.linear2 = nn.Dense(int(self.num_kernels))
        self.linear3 = nn.Dense(int(self.num_kernels / 2))
        self.linear4 = nn.Dense(self.out_features)

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """

        # linear layers
        out1 = self.linear1(x)
        out2 = self.linear2(nn.relu(out1))
        out3 = self.linear3(nn.relu(out2))
        out = self.linear4(nn.relu(out3))

        return out


class ClusterWCRBFNet(nn.Module):
    in_features: int
    out_features: int
    num_kernels: int
    basis_func: Callable
    num_regions: int
    # TODO: centers to be added

    def setup(self):
        # check type of layers
        # if self.centers is not None:
        #     if self.fixed_centers:
        #         if self.fixed_width:
        #             layer_fn = FixedWidthCenteredRBFLayer
        #         else:
        #             layer_fn = FixedCenteredRBFLayer
        #     else:
        #         layer_fn = WarmstartCenteredRBFLayer
        # else:
        #     layer_fn = RBFLayer

        layer_fn = RBFLayer

        # instantiate multi-headed RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            layer_fn,
            in_axes=None,
            out_axes=0,
            axis_size=self.num_regions,
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )

        # Batching RBFNets via vmap
        broadcasted_rbf = nn.vmap(
            broadcasted_rbf,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.rbf_list = broadcasted_rbf(
            in_features=self.in_features,
            num_kernels=self.num_kernels,
            basis_func=self.basis_func,
        )
        self.linear = nn.Dense(self.out_features)
        self.cluster = nn.Dense(self.num_regions)

    def __call__(self, x):
        """
        Forward,

        Args:
            x (input vector, jnp.DeviceArray (batch_size, in_features))
        """
        # rbf networks
        all_x = self.rbf_list(x)

        # clustering, output is prob for each region to activate
        logits = self.cluster(x)
        cluster_ind = nn.softmax(logits)
        cluster_ind_rep = jnp.repeat(
            jnp.expand_dims(cluster_ind, -1), self.num_kernels, axis=-1
        )

        # interpolation
        rbf_out = jnp.sum(cluster_ind_rep * all_x, axis=1)

        # linear layer
        out = self.linear(rbf_out)

        return out, logits
