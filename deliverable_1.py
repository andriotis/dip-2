import numpy as np
from tqdm import tqdm
import typing as tp
from skimage import io, color, feature
import matplotlib.pyplot as plt
import seaborn as sns


def my_hough_transform(
    img_binary: np.ndarray, d_rho: int, d_theta: float, n: int
) -> tp.Tuple[np.ndarray, np.ndarray, int]:
    """
    Perform the Hough Transform on a binary image to detect straight lines.

    Parameters:
    img_binary (np.ndarray): A 2D numpy array representing the binary image where edge points are marked.
    d_rho (int): The distance resolution of the accumulator in pixels.
    d_theta (float): The angle resolution of the accumulator in radians.
    n (int): The number of top lines to detect.

    Returns:
    tp.Tuple[np.ndarray, np.ndarray, int]:
        - H (np.ndarray): The Hough accumulator array.
        - lines (np.ndarray): An array of the top `n` lines represented by tuples of (rho, theta).
        - res (int): The count of pixels that do not belong to the top `n` lines.

    Notes:
    The Hough Transform is a feature extraction technique used in image analysis,
    computer vision, and digital image processing. The purpose of the technique is
    to find imperfect instances of objects within a certain class of shapes by a voting procedure.

    The algorithm proceeds as follows:
    1. Calculate the maximum possible value of rho based on the image dimensions.
    2. Create the rho and theta ranges based on `d_rho` and `d_theta`.
    3. Initialize the Hough accumulator array `H`.
    4. For each edge point in the binary image, compute the corresponding rho for each theta and update the accumulator.
    5. Identify the top `n` lines by finding the highest values in the accumulator.
    6. Track pixels that belong to these top `n` lines.
    7. Count the number of edge pixels that do not belong to any of the top `n` lines.

    Example:
    --------
    >>> import numpy as np
    >>> from my_hough_transform import my_hough_transform
    >>> binary = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0]])
    >>> d_rho = 1
    >>> d_theta = np.pi / 180
    >>> n = 1
    >>> H, lines, res = my_hough_transform(binary, d_rho, d_theta, n)
    >>> print(H)
    >>> print(lines)
    >>> print(res)
    """
    # Get the dimensions of the input binary image
    height, width = img_binary.shape

    # Calculate the maximum possible value of rho
    max_rho = np.sqrt(height**2 + width**2)
    rhos = np.arange(-max_rho, max_rho, d_rho)
    thetas = np.arange(-np.pi, np.pi, d_theta)

    # Create the Hough accumulator array
    H = np.zeros((len(rhos), len(thetas)))

    # Get the indices of the edge points in the binary image
    y_idxs, x_idxs = np.nonzero(img_binary)

    # Perform the Hough transform
    for i in tqdm(
        range(len(x_idxs)),
        desc="Performing Hough Transform",
        unit="pixels",
        leave=False,
    ):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhos - rho))
            H[rho_idx, t_idx] += 1

    # Find the top n lines
    temp_H = np.copy(H)
    lines = []
    for i in tqdm(range(n), desc="Finding Top Lines", unit="lines", leave=False):
        idx = np.argmax(temp_H)
        rho_idx, theta_idx = np.unravel_index(idx, temp_H.shape)
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))
        temp_H[rho_idx, theta_idx] = 0  # Reset the maximum value to find the next one
        temp_H[rho_idx - 10 : rho_idx + 10, theta_idx - 10 : theta_idx + 10] = (
            0  # Zero out the neighborhood of the maximum value
        )

    # Create a set to track pixels that belong to the top n lines
    pixels_on_top_lines = set()

    # Identify pixels that belong to the top n lines
    for rho, theta in tqdm(
        lines, desc="Identifying Pixels on Top Lines", unit="lines", leave=False
    ):
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            if np.isclose(rho, x * np.cos(theta) + y * np.sin(theta), atol=d_rho):
                pixels_on_top_lines.add((x, y))

    # Count the number of pixels that do not belong to the top n lines
    all_pixels = set(zip(x_idxs, y_idxs))
    pixels_not_on_top_lines = all_pixels - pixels_on_top_lines
    res = len(pixels_not_on_top_lines)

    return H, np.array(lines), res


if __name__ == "__main__":
    original = io.imread("data/im2.jpg")
    gray = color.rgb2gray(original)
    binary = feature.canny(gray, sigma=5)
    d_rho = 5
    d_theta = np.pi / 180
    n = 10
    H, L, res = my_hough_transform(binary, d_rho, d_theta, n)
    # Get the dimensions of the input binary image
    height, width = binary.shape

    # Calculate the maximum possible value of rho
    max_rho = np.sqrt(height**2 + width**2)
    rhos = np.arange(-max_rho, max_rho, d_rho)
    thetas = np.arange(-np.pi, np.pi, d_theta)
    # Plot the Hough accumulator array
    ax = sns.heatmap(H, cmap="viridis")
    ax.set_title("Hough Accumulator Array")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Distance from the Origin")

    ax.set_xticks(np.arange(0, len(thetas), len(thetas) // 10))
    ax.set_xticklabels(np.round(thetas[:: len(thetas) // 10], 2))

    ax.set_yticks(np.arange(0, len(rhos), len(rhos) // 10))
    ax.set_yticklabels(np.round(rhos[:: len(rhos) // 10], 2))

    # Scatter rho, theta pairs
    for rho, theta in L:
        # add different color for each line to make it easier to distinguish NOT RANDOM
        color = np.random.rand(
            3,
        )
        ax.scatter(
            thetas.tolist().index(theta),
            rhos.tolist().index(rho),
            c=[color],
            s=100,
            marker="x",
        )

    plt.figure(figsize=(10, 10))
    for rho, theta in L:
        color = np.random.rand(
            3,
        )
        p0 = np.array([rho * np.cos(theta), rho * np.sin(theta)])
        direction = np.array([p0[1], -p0[0]]) / np.linalg.norm(p0)
        p1 = p0 + 5000 * direction
        p2 = p0 - 5000 * direction
        # plt.plot(p0[0], p0[1], "ro")
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c=color)
    plt.imshow(original, cmap="gray")
    plt.title(f"Top {len(L)} Lines")
    plt.show()
