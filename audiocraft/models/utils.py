import torch
import numpy as np

def decimal_to_ternary_matrix(decimals, D):
    """
    Convert a tensor of decimal numbers to a D*T ternary matrix for each batch.

    Arguments
    ---------
    decimals : torch.Tensor
        A 2D tensor of decimal numbers with shape (B, T), where B is the batch size
        and T is the number of elements in each batch.
    D : int
        Number of ternary digits to represent each number (depth).

    Returns
    -------
    torch.Tensor
        A 3D tensor of shape (B, D, T) where each slice along the first dimension
        corresponds to a batch, and each column is represented as a ternary number.
    """
    B, T = decimals.shape
    ternary_matrix = torch.zeros((B, D, T), dtype=torch.long)
    for pos in range(D):
        ternary_matrix[:, pos, :] = decimals % 3  # Modulo operation
        decimals //= 3  # Floor division for next ternary digit

    return ternary_matrix


def ternary_matrix_to_decimal(matrix):
    """
    Convert a B*D*N ternary matrix to a 2D array of decimal numbers for each batch.

    Arguments
    ---------
    matrix : numpy.ndarray
        A 3D numpy array of shape (B, D, N), where B is the batch size, D is the number
        of ternary digits, and N is the number of ternary numbers in each batch.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (B, N), where each value represents the decimal
        equivalent of the corresponding ternary number in the input matrix.
    """
    (
        B,
        D,
        N,
    ) = (
        matrix.shape
    )  # B is the batch size, D is the number of digits, N is the number of ternary numbers
    powers_of_three = 3 ** np.arange(D)  # [3^0, 3^1, ..., 3^(D-1)]

    # Reshape powers_of_three for broadcasting: [D] -> [1, D, 1]
    powers_of_three = powers_of_three[:, np.newaxis]  # Shape [D, 1]

    # Compute dot product using broadcasting: matrix * powers_of_three along D axis
    decimals = np.sum(matrix * powers_of_three, axis=1)  # Sum along the D axis

    return decimals


def get_padding(kernel_size, dilation=1):
    """
    Computes the padding size for a given kernel size and dilation.

    Arguments
    ---------
    kernel_size : int
        Size of the convolutional kernel.
    dilation : int, optional
        Dilation factor for convolution (default is 1).

    Returns
    -------
    int
        Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)