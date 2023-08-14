import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    image_left = cv2.imread("../../BPARR_latex/Graphics/dataset_original.png")
    image_right = cv2.imread("../../BPARR_latex/Graphics/dataset_mask.png")

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB))
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY))
    ax[1].axis("off")

    plt.savefig("./pokus.png", dpi=600, bbox_inches="tight", pad_inches=0.0, transparent=True)

    plt.show()