#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The code was imported & slightly modified from
https://github.com/pfnet-research/chainer-graph-cnn
"""

import chainer
import chainer.functions as F
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer import reporter

from chainer_chemistry.utils import coarsening
from chainer_chemistry.functions.pooling.spectral_graph_max_pooling import SpectralGraphMaxPoolingFunction  # NOQA
from chainer_chemistry.links.connection.spectral_graph_convolution import SpectralGraphConvolution  # NOQA


class SpectralGraphCNN(chainer.Chain):
    """Graph CNN example implementation.

    Uses the GC32-P4-GC64-P4-FC512 architecture as in the original paper.
    """

    def __init__(self, A, n_out=10):
        super(SpectralGraphCNN, self).__init__()

        # Precompute the coarsened graphs
        graphs, pooling_inds = coarsening.coarsen(A, levels=4)
        # In order to simulate 2x2 max pooling, combine the 4 levels
        # of graphs into 2 levels by combining pooling indices.
        graphs, pooling_inds = coarsening.combine(graphs, pooling_inds, 2)

        graph_conv_layers_list = []

        # sizes for graph conv layers
        graph_pool_list = []
        sizes = [32, 64]
        for i, (g, inds, s) in enumerate(zip(graphs, pooling_inds, sizes)):
            f = SpectralGraphConvolution(None, s, g, K=25)
            p = SpectralGraphMaxPoolingFunction(inds)
            graph_conv_layers_list.append(f)
            graph_pool_list.append(p)

        # sizes for linear layers
        sizes = [512]
        with self.init_scope():
            self.graph_conv_layers = chainer.ChainList(
                *graph_conv_layers_list)
            self.linear_layers = chainer.ChainList(
                *[L.Linear(None, s) for s in sizes])
            self.cls_layer = L.Linear(None, n_out)
        self.graph_pool_functions = graph_pool_list

    def __call__(self, x, *args):
        # x.shape = (n_batch, n_channels, h*w)
        dropout_ratio = 0.5

        h = x
        # Graph convolutional layers
        for f, p in zip(self.graph_conv_layers, self.graph_pool_functions):
            h = p(F.relu(f(h)))

        # Fully connected layers
        for f in self.linear_layers:
            h = F.relu(F.dropout(f(h), dropout_ratio))

        # Linear classification layer
        h = self.cls_layer(h)

        if args:
            labels = args[0]
            loss = F.softmax_cross_entropy(h, labels)
            acc = accuracy.accuracy(h, labels)
            reporter.report({
                'loss': loss,
                'accuracy': acc},
                self)

            return loss

        return h
