from pathlib import Path
from BPARR_practical.comparison.comparison import load_camera_intrinsic
from BPARR_practical.comparison.comparison import generate_homographic_trans_mtx, create_steerable_filters
import cv2
from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":
    source = Path("../resources/jetbot_datasets/dataset_4/000005.png")
    destination = Path("./latex_gen/calibration_pic.png")
    destination.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(source.__str__())

    # load undistortion parameters
    params = load_camera_intrinsic("../camera_intrinsic_calibration.json")
    # transfer parameters from dictionary into parameters for undistortion
    mtx = params["mtx"]
    refined_mtx = params["refined_mtx"]
    dist = params["dist"]
    roi = params["roi"]
    ## specify roi cutting --> need to be adjusted if source images are different from calibration dataset
    x, y, w, h = roi

    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, refined_mtx, (image.shape[1], image.shape[0]), cv2.CV_32FC1)
    im_undist = cv2.remap(image, map1, map2,
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))[y:y + h, x:x + w]
    H = generate_homographic_trans_mtx(im_undist)
    im_undist2 = cv2.warpPerspective(im_undist, H, (w, h))

    fig, ax = plt.subplots(2,2)
    ax[0, 0].imshow(cv2.cvtColor(im_undist, cv2.COLOR_BGR2RGB))
    ax[0, 0].axis("off")
    ax[0, 0].set_title("(a)", y=-0.2)

    ax[0, 1].imshow(cv2.cvtColor(im_undist2, cv2.COLOR_BGR2RGB))
    ax[0, 1].axis("off")
    ax[0, 1].set_title("(b)", y=-0.2)

    image2 = cv2.imread("../../BPARR_latex/Graphics/algorithmic_input.png")
    # im2_undist = cv2.remap(image2, map1, map2,
    #                       interpolation=cv2.INTER_CUBIC,
    #                       borderMode=cv2.BORDER_CONSTANT,
    #                       borderValue=(0, 0, 0))[y:y + h, x:x + w]
    im2_undist = image2
    im2_undist2 = cv2.warpPerspective(im2_undist, H, (w, h))

    ax[1, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    ax[1, 0].axis("off")
    ax[1, 0].set_title("(c)", y=-0.2)

    ax[1, 1].imshow(cv2.cvtColor(im2_undist2, cv2.COLOR_BGR2RGB))
    ax[1, 1].axis("off")
    ax[1, 1].set_title("(d)", y=-0.2)
    # plt.savefig("./pokus.png", dpi=600, bbox_inches="tight", pad_inches=0.0)
    plt.show()

    # convert image to grayscale for edge detection
    img_gray = cv2.cvtColor(im2_undist2, cv2.COLOR_BGR2GRAY)

    # gaussian blurring
    kernel = np.ones((10, 10), np.float32) / 100
    img_gray = cv2.filter2D(img_gray, -1, kernel)

    # create steerable filter
    fltr_rl = create_steerable_filters(kernel_size=10, variance=(8, 1), center=(0, 0))
    fltr_lr = fltr_rl[:, ::-1]

    # feature extraction
    filter_im = cv2.filter2D(src=img_gray, ddepth=-1, kernel=fltr_rl)
    filter_im1 = cv2.filter2D(src=img_gray, ddepth=-1, kernel=fltr_lr)

    # binary thresholding
    thr_bin, filter_im_bin_rl = cv2.threshold(filter_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter_im_bin_rl = filter_im_bin_rl.astype(dtype="int")
    filter_im_bin_lr = ((filter_im1 >= thr_bin) * 255).astype(dtype="int")
    filter_im_bin = (((filter_im_bin_rl + filter_im_bin_lr) >= 255) * 255).astype(dtype="uint8")

    cv2.imwrite("../algorithmic_bin.png", filter_im_bin)

    cv2.imshow("pokus", filter_im_bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
