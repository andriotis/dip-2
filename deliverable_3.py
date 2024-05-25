import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm


def my_img_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a given angle using bilinear interpolation.

    Parameters:
    img (np.ndarray): The input image to be rotated. Can be either grayscale or RGB.
    angle (float): The rotation angle in radians.

    Returns:
    np.ndarray: The rotated image with a black background.

    Notes:
    This function rotates an image by a specified angle counterclockwise. The size of the output image
    is adjusted to ensure that the entire rotated image fits within the new dimensions. Bilinear
    interpolation is used to calculate the pixel values of the rotated image.

    The new dimensions of the rotated image are calculated based on the original image dimensions and
    the rotation angle to ensure that the entire image is captured without cropping.
    """
    # Get the dimensions of the input image
    height, width = img.shape[:2]

    # Calculate the rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = -np.sin(angle)

    # Calculate the new bounding box dimensions
    new_width = int(abs(height * sin_theta) + abs(width * cos_theta))
    new_height = int(abs(height * cos_theta) + abs(width * sin_theta))

    # Create the output image with black background
    if len(img.shape) == 3:
        rot_img = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
    else:
        rot_img = np.zeros((new_height, new_width), dtype=img.dtype)

    # Calculate the center of the original and new image
    center_x, center_y = width // 2, height // 2
    new_center_x, new_center_y = new_width // 2, new_height // 2

    # Perform the rotation
    for i in tqdm(
        range(new_height), desc=f"Rotating image by {angle * 180 / np.pi} degrees"
    ):
        for j in range(new_width):
            # Calculate the coordinates in the original image
            x = (
                (j - new_center_x) * cos_theta
                + (i - new_center_y) * sin_theta
                + center_x
            )
            y = (
                -(j - new_center_x) * sin_theta
                + (i - new_center_y) * cos_theta
                + center_y
            )

            if 0 <= x < width and 0 <= y < height:
                # Bilinear interpolation
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)

                a = x - x1
                b = y - y1

                if len(img.shape) == 3:  # RGB image
                    for c in range(img.shape[2]):
                        rot_img[i, j, c] = (
                            (1 - a) * (1 - b) * img[y1, x1, c]
                            + a * (1 - b) * img[y1, x2, c]
                            + (1 - a) * b * img[y2, x1, c]
                            + a * b * img[y2, x2, c]
                        )
                else:  # Grayscale image
                    rot_img[i, j] = (
                        (1 - a) * (1 - b) * img[y1, x1]
                        + a * (1 - b) * img[y1, x2]
                        + (1 - a) * b * img[y2, x1]
                        + a * b * img[y2, x2]
                    )

    return rot_img


if __name__ == "__main__":

    print("Reading image...")
    original = io.imread("data/im2.jpg")
    angle_54 = (np.pi / 180) * 54  # 54 degrees in radians
    angle_213 = (np.pi / 180) * 213  # 213 degrees in radians

    rotated_img_54 = my_img_rotation(original, angle_54)
    rotated_img_213 = my_img_rotation(original, angle_213)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(rotated_img_54, cmap="gray")
    plt.title("54 Degrees")
    plt.subplot(1, 2, 2)
    plt.imshow(rotated_img_213, cmap="gray")
    plt.title("213 Degrees")
    plt.show()
