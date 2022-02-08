from skimage import io
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image = io.imread("assets/color.png")

    print("This is going to take a while...")
    for row in image:
        for pix in row:
            # RGB range is 0..255 here
            r, g, b = pix[0], pix[1], pix[2]
            # I'm going to assume that we are allowed to use the rgb2hsv function
            h, s, v = rgb2hsv(pix)

            # Only show red/orange colors and grayscale the rest
            # Number based on guesswork
            # TODO: Finetune this, or maybe some other selection method.
            if h > 0.14:
                # Based on https://en.wikipedia.org/wiki/Grayscale
                y = r * 0.2126 + g * 0.7152 + b * 0.0722
                pix.fill(y)

    plt.imshow(image)
    plt.show()
