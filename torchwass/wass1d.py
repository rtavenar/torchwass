import torch
from torchsearchsorted import searchsorted

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def torch_diff(a):
    return a[1:] - a[:-1]


def wasserstein(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute Wass_p between two one-dimensional distributions :math:`u` and
    :math:`v`.

    This implementation is an adaptation of the scipy implementation of
    `scipy.stats._cdf_distance` for torch tensors.

    Parameters
    ----------
    u_values, v_values : torch tensors of shape (ns, ) and (nt, )
        Values observed in the (empirical) distributions.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.
    Returns
    -------
    distance : float
        The computed distance between the distributions.
    """
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)

    all_values = torch.cat((u_values, v_values))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = torch_diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = searchsorted(u_values[u_sorter], all_values[:-1], 'right')
    v_cdf_indices = searchsorted(v_values[v_sorter], all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size  # TODO
    else:
        u_sorted_cumweights = torch.cat(([0], np.cumsum(u_weights[u_sorter])))  # TODO
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size  # TODO
    else:
        v_sorted_cumweights = torch.cat(([0], np.cumsum(v_weights[v_sorter])))  # TODO
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    return torch.sum(torch.mul(torch.abs(u_cdf - v_cdf),
                               torch.pow(deltas, p)))