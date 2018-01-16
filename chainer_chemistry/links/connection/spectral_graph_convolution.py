#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The code was imported & slightly modified from
https://github.com/pfnet-research/chainer-graph-cnn
"""

import numpy
from scipy import sparse

from chainer import cuda, variable
from chainer import initializers
from chainer import link

from chainer_chemistry.functions.connection.spectral_graph_convolution import SpectralGraphConvolutionFunction  # NOQA


def create_laplacian(W, normalize=True):
    n = W.shape[0]
    W = sparse.csr_matrix(W)
    WW_diag = W.dot(sparse.csr_matrix(numpy.ones((n, 1)))).todense()
    if normalize:
        WWds = numpy.sqrt(WW_diag)
        # Let the inverse of zero entries become zero.
        WWds[WWds == 0] = numpy.float("inf")
        WW_diag_invroot = 1. / WWds
        D_invroot = sparse.lil_matrix((n, n))
        D_invroot.setdiag(WW_diag_invroot)
        D_invroot = sparse.csr_matrix(D_invroot)
        I = sparse.identity(W.shape[0], format='csr', dtype=W.dtype)
        L = I - D_invroot.dot(W.dot(D_invroot))
    else:
        D = sparse.lil_matrix((n, n))
        D.setdiag(WW_diag)
        D = sparse.csr_matrix(D)
        L = D - W

    return L.astype(W.dtype)


class SpectralGraphConvolution(link.Link):
    """Graph convolutional layer.

    This link wraps the :func:`spectral_graph_convolution` function and holds the filter
    weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        A (~ndarray): Weight matrix describing the graph.
        K (int): Polynomial order of the Chebyshev approximation.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
       See :func:`spectral_graph_convolution` for the definition of
       graph convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    Graph convolutional layer using Chebyshev polynomials
    in the graph spectral domain.

    This link implements the graph convolution described in
    the paper

    Defferrard et al. "Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering", NIPS 2016.

    """

    def __init__(self, in_channels, out_channels, A, K, bias=0,
                 nobias=False, initialW=None, initial_bias=None):
        super(SpectralGraphConvolution, self).__init__()

        L = create_laplacian(A)

        self.K = K
        self.out_channels = out_channels

        with self.init_scope():
            self._W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(self._W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = bias
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

        self.func = SpectralGraphConvolutionFunction(L, K)

    def to_cpu(self):
        super(SpectralGraphConvolution, self).to_cpu()
        self.func.to_cpu()

    def to_gpu(self, device=None):
        with cuda.get_device_from_id(device):
            super(SpectralGraphConvolution, self).to_gpu(device)
            self.func.to_gpu(device)

    def _initialize_params(self, in_channels):
        W_shape = (self.out_channels, in_channels, self.K)
        self.W.initialize(W_shape)

    def __call__(self, x):
        """Applies the graph convolutional layer.

        Args:
            x: (~chainer.Variable): Input graph signal.

        Returns:
            ~chainer.Variable: Output of the graph convolution.
        """
        if self.W.data is None:
            with cuda.get_device_from_id(self._device_id):
                self._initialize_params(x.shape[1])
        if self.b is None:
            return self.func(x, self.W)
        else:
            return self.func(x, self.W, self.b)
