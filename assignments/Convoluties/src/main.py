import matplotlib.pyplot as plt
from skimage import data, filters, feature
from scipy import ndimage

if __name__ == "__main__":
    image = data.camera()

    # Blur image
    blur_filter = [[1 / 9, 1 / 9, 1 / 9]] * 3
    blur_image = ndimage.convolve(image, blur_filter)
    blur_image = ndimage.convolve(blur_image, blur_filter)

    # Positive Laplacian Operator
    laplacian_filter = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]

    laplacian_image = ndimage.convolve(blur_image, laplacian_filter)

    # Sobel Horizontal
    sobel_h_filter = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
    sobel_v_filter = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]
    sobel_h_image = ndimage.convolve(blur_image, sobel_h_filter)
    sobel_v_image = ndimage.convolve(blur_image, sobel_v_filter)

    # Scharr
    scharr_h_filter = [
        [47, 0, -47],
        [162, 0, -162],
        [47, 0, -47],
    ]
    scharr_h_image = ndimage.convolve(blur_image, scharr_h_filter)

    # Using skimage filters
    sk_roberts = filters.roberts(image)
    sk_sobel = filters.sobel(image)
    sk_scharr = filters.scharr(image)

    # Canny filter
    canny_default_image = feature.canny(image)

    canny_image = feature.canny(
        image, sigma=1.4, low_threshold=0.4 * 255, high_threshold=0.5 * 255
    )

    # Show results
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        nrows=3, ncols=3
    )

    ax1.imshow(laplacian_image, cmap="gray")
    ax1.set_title("Positive Laplacian Operator")

    ax2.imshow(sobel_h_image, cmap="gray")
    ax2.set_title("Sobel Horizontal Operator")

    ax3.imshow(scharr_h_image, cmap="gray")
    ax3.set_title("Scharr Operator")

    ax4.imshow(sk_roberts, cmap="gray")
    ax4.set_title("(SKI Filter) Roberts")

    ax5.imshow(sk_sobel, cmap="gray")
    ax5.set_title("(SKI Filter) Sobel")

    ax6.imshow(sk_scharr, cmap="gray")
    ax6.set_title("(SKI Filter) Scharr")

    ax7.imshow(canny_default_image, cmap="gray")
    ax7.set_title("Canny (Default Params)")

    ax8.imshow(canny_image, cmap="gray")
    ax8.set_title("Canny (Tweaked Params)")

    plt.show()
