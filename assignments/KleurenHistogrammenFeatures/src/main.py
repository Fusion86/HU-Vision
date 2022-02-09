import numpy as np
from skimage import io
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = io.imread("assets/color.png")
    image_gray = image.copy()

    # Which color range to keep.
    # Random math has been added to convert a photoshop range to skimage range.
    min_h = 24 / 360
    max_h = 45 / 360

    print("This is going to take a while...")
    for y in range(image_gray.shape[0]):
        for x in range(image_gray.shape[1]):
            # RGB range is 0..255
            pix = image_gray[y, x]
            r, g, b = pix[0], pix[1], pix[2]
            # HSV range is 0..1
            # I'm going to assume that we are allowed to use the rgb2hsv function.
            h, s, v = rgb2hsv(pix)

            # Only show yellow colors and grayscale the rest.
            # Number based on guesswork.
            if h < min_h or h > max_h:
                # Based on https://en.wikipedia.org/wiki/Grayscale
                yp = r * 0.2126 + g * 0.7152 + b * 0.0722
                pix.fill(yp)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.imshow(image)
    ax2.imshow(image_gray)

    n_bins = 10
    hist_range = (0, 1)
    # Based on https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_rgb_to_hsv.html
    hue_image = rgb2hsv(image)[:, :, 0].flatten()
    hue_image_gray = rgb2hsv(image_gray)[:, :, 0].flatten()

    ax3.hist(hue_image, n_bins, hist_range)
    ax3.set_title("Original Hue")
    ax4.hist(hue_image_gray, n_bins, hist_range)
    ax4.set_title("Grayscale Hue")

    plt.show()
