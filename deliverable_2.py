import numpy as np
from scipy.signal import convolve2d
from skimage import io, color, feature
import matplotlib.pyplot as plt


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a Gaussian kernel.

    Parameters:
    size (int): The size of the kernel. It should be an odd number.
    sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    np.ndarray: A 2D array representing the Gaussian kernel.

    Notes:
    The Gaussian kernel is used to smooth images and is defined by the
    Gaussian function. The function uses a lambda to generate a 2D kernel
    with the given size and standard deviation.

    Example:
    --------
    >>> size = 5
    >>> sigma = 1.0
    >>> kernel = gaussian_kernel(size, sigma)
    >>> print(kernel)
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)
        ),
        (size, size),
    )
    return kernel / np.sum(kernel)


def harris_response(img: np.ndarray, k: float, sigma: float) -> np.ndarray:
    """
    Compute the Harris corner response for an image.

    Parameters:
    img (np.ndarray): A 2D array representing the grayscale image.
    k (float): The sensitivity factor to separate corners from edges, typically between 0.04 and 0.06.
    sigma (float): The standard deviation for the Gaussian smoothing.

    Returns:
    np.ndarray: A 2D array representing the Harris corner response.

    Notes:
    The Harris corner detector is an algorithm to detect corners in images.
    It involves computing the image gradients, constructing a Gaussian
    weighted sum of the gradient products, and then computing the Harris
    response which highlights the corners.

    Example:
    --------
    >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    >>> k = 0.05
    >>> sigma = 1.0
    >>> response = harris_response(img, k, sigma)
    >>> print(response)
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Compute the image gradients
    I_x = convolve2d(img, sobel_x, mode="same")
    I_y = convolve2d(img, sobel_y, mode="same")

    # Gaussian kernel
    size = int(4 * sigma)  # typically 3-sigma rule
    gaussian_kernel_ = gaussian_kernel(size, sigma)

    # Compute the elements of the structure tensor
    I_x2 = convolve2d(I_x**2, gaussian_kernel_, mode="same")
    I_y2 = convolve2d(I_y**2, gaussian_kernel_, mode="same")
    I_xy = convolve2d(I_x * I_y, gaussian_kernel_, mode="same")

    # Compute the Harris response
    det = I_x2 * I_y2 - I_xy**2
    trace = I_x2 + I_y2
    response = det - k * trace**2

    return response


def corner_locations(harris_response: np.ndarray, rel_threshold: float) -> np.ndarray:
    """
    Find corner locations in the Harris response map.

    Parameters:
    harris_response (np.ndarray): A 2D array representing the Harris corner response.
    rel_threshold (float): The relative threshold to identify strong corners. It is a fraction of the maximum response value.

    Returns:
    np.ndarray: An array of coordinates where corners are detected.

    Notes:
    This function identifies corner points in the Harris response map by applying
    a relative threshold. Only points with response values above the threshold are
    considered as corners.

    Example:
    --------
    >>> harris_resp = np.array([[0.1, 0.2, 0.1], [0.4, 0.8, 0.4], [0.1, 0.2, 0.1]])
    >>> rel_threshold = 0.5
    >>> corners = corner_locations(harris_resp, rel_threshold)
    >>> print(corners)
    """
    # Compute the threshold
    threshold = rel_threshold * harris_response.max()
    # Find corners by applying a threshold
    corner_locations = harris_response > threshold
    # Find the indices of the corners
    corner_locations = np.argwhere(corner_locations)
    return corner_locations


if __name__ == "__main__":
    print("Reading image...")
    original = io.imread("data/im2.jpg")
    gray = color.rgb2gray(original)
    binary = feature.canny(gray, sigma=5)

    k = 0.04
    sigma = 3.0
    print("Detecting corners...")
    response = harris_response(gray, k, sigma)
    rel_threshold = 0.1
    print("Finding corner locations...")
    corners = corner_locations(response, rel_threshold)

    plt.figure(figsize=(10, 10))
    plt.imshow(gray, cmap="gray")
    plt.scatter(corners[:, 1], corners[:, 0], c="r", s=10)
    plt.title("Harris Corners")
    plt.show()
