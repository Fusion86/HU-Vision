import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, io
from skimage import img_as_float


def rotationMatrix(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


if __name__ == "__main__":
    image = io.imread("zorse.jpg")
    img = img_as_float(image)
    w, h = img.shape[:2]

    # Matrix spec
    # [i,0,x]
    # [0,i,y]
    # [0,0,i]
    # i = scale x,y. 1 -> don't scale (identity)
    # x,y = translate in px. 0 -> don't translate

    # Copied from docs:
    # Now let’s apply this transformation to an image.
    # Because we are trying to reconstruct the image after transformation, it is not useful to see where a coordinate from the input image ends up in the output,
    # which is what the transform gives us. Instead, for every pixel (coordinate) in the output image, we want to figure out where in the input image it comes from.
    # Therefore, we need to use the inverse of tform, rather than tform directly.

    # Move and rotate
    matrix_tl = np.array(
        [
            [1, 0, 70],
            [0, 1, -150],
            [0, 0, 1],
        ]
    )
    matrix_rot = rotationMatrix(np.pi / 6)
    matrix = np.dot(matrix_rot, matrix_tl)

    tform_tlrot = transform.AffineTransform(matrix=matrix)
    tf_img = transform.warp(img, tform_tlrot.inverse)

    # Shear
    tform_shear = transform.AffineTransform(shear=np.pi / 6)
    tf_img_shear = transform.warp(img, tform_shear.inverse)

    # Stretch + center vertically
    matrix = np.array(
        [
            [1, 0, 0],
            [0, 3, -(h / 2)],
            [0, 0, 1],
        ]
    )
    tform = transform.AffineTransform(matrix=matrix)
    tf_img_stretch = transform.warp(img, tform.inverse)

    # Combined
    # tf_img_comb = transform.warp(img, tform_tlrot.inverse)
    # tf_img_comb = transform.warp(tf_img_comb, tform_shear.inverse)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.imshow(image)
    ax1.set_title("A zorse")
    ax2.imshow(tf_img)
    ax2.set_title("Move and rotate")
    ax3.imshow(tf_img_shear)
    ax3.set_title("Shear")
    ax4.imshow(tf_img_stretch)
    ax4.set_title("Stretch and center (vert)")

    plt.show()
