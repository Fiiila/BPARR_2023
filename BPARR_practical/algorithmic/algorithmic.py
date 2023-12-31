from pathlib import Path
import glob
import cv2
import numpy as np
from BPARR_practical.preprocessing import generate_undistortion_maps, load_camera_intrinsic, \
    generate_homographic_trans_mtx
from BPARR_practical.postprocessing import find_line_begining, cluster_with_sliding_window, compute_line_polynom


def create_steerable_filters(kernel_size, variance=(2, 2), center=(0, 0), amplitude=1):
    """Function to generate 2D Gaussian kernel derived in x direction

    :param np.ndarray kernel_size: size of 2D kernel matrix
    :param tuple[int, int] variance: of 2D Gaussian in axes (x, y)
    :param tuple[int, int] center: center of 2D Gaussian (x, y)
    :param int amplitude: amplitude of 2D Gaussian
    :return np.ndarray: returns 2D matrix of derived 2D Gaussian in x axe
    """
    # 2D Gaussian creation
    x = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    y = x.copy()
    x0, y0 = center
    varx, vary = variance
    gauss_kernel = np.zeros((len(y), len(x)))
    gauss_kernel_diff = np.zeros((len(y), len(x)))
    # computing 2D Gaussian and its derivative form in x direction
    for idy in range(len(y)):
        for idx in range(len(x)):
            gauss_kernel[idy, idx] = amplitude * np.exp(
                -((np.square(x[idx] - x0) / np.square(2 * varx)) + (np.square(y[idy] - y0) / np.square(2 * vary))))
            gauss_kernel_diff[idy, idx] = -(
                    np.exp(-np.square(x[idx] - x0) / (2 * varx) - np.square(y[idy] - y0) / (2 * vary)) * (
                    2 * x[idx] - 2 * x0)) / (2 * varx)

    # Alternative method to get derivative form of 2D Gaussian
    # tmp = np.gradient(gauss_kernel)
    return gauss_kernel_diff


def algorithmic_extraction(input_image, homograph_mtx, w, h):
    # warp perspective
    img_warped = cv2.warpPerspective(input_image, homograph_mtx, (w, h))

    # convert image to grayscale for edge detection
    img_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    # gaussian blurring
    kernel = np.ones((10, 10), np.float32) / 100
    img_gray = cv2.filter2D(img_gray, -1, kernel)

    # feature extraction
    filter_im = cv2.filter2D(src=img_gray, ddepth=-1, kernel=fltr_rl)
    filter_im1 = cv2.filter2D(src=img_gray, ddepth=-1, kernel=fltr_lr)

    # binary thresholding
    thr_bin, filter_im_bin_rl = cv2.threshold(filter_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter_im_bin_rl = filter_im_bin_rl.astype(dtype="int")
    filter_im_bin_lr = ((filter_im1 > thr_bin) * 255).astype(dtype="int")
    filter_im_bin = (((filter_im_bin_rl + filter_im_bin_lr) >= 255) * 255).astype(dtype="uint8")

    # mask out unwanted edges
    cutout_width = 30  # number of px, that will be extracted from middle to move closer...needs to be odd
    edge_mask = np.zeros((h, w), dtype="uint8")
    mask = ((img_gray > 0) * 1).astype(dtype="uint8")
    edge_mask[:, cutout_width // 2:w // 2] += mask[:, 0:w // 2 - cutout_width // 2]
    edge_mask[:, w // 2:w - cutout_width // 2] += mask[:, w // 2 + cutout_width // 2:w]
    filter_im_bin *= edge_mask

    return filter_im_bin


if __name__ == "__main__":
    visualise = True
    input_images = Path("../resources/jetbot_datasets/dataset_10")

    # Load created camera calibration parameters
    calib_path = Path("../camera_intrinsic_calibration.json")
    params = load_camera_intrinsic(calib_path=calib_path)
    # transfer parameters from dictionary into parameters for undistortion
    mtx = params["mtx"]
    refined_mtx = params["refined_mtx"]
    dist = params["dist"]
    roi = params["roi"]
    # specify roi cutting --> need to be adjusted if source images are different from calibration dataset
    x, y, w, h = roi

    # Load input image paths into memory
    images = glob.glob(input_images.joinpath("*.png").__str__())

    # Generate maps for undistortion
    h_im, w_im = cv2.imread(images[0]).shape[:2]
    mapx, mapy = generate_undistortion_maps(mtx, dist, refined_mtx, w_im, h_im)

    # generate homographic transformation matrix
    img_undist = cv2.undistort(src=cv2.imread("../resources/jetbot_datasets/dataset_4/000000.png"), cameraMatrix=mtx,
                               distCoeffs=dist, dst=None, newCameraMatrix=refined_mtx)
    img_crop = img_undist[y:y + h, x:x + w]
    H = generate_homographic_trans_mtx(img=img_crop)

    # Create steerable filter
    fltr_rl = create_steerable_filters(kernel_size=10, variance=(8, 1), center=(0, 0))
    fltr_lr = fltr_rl[:, ::-1]

    # Cycle for line extraction
    for image_number, image in enumerate(images):
        # Load raw image
        img_raw = cv2.imread(image)

        # Undistortion input image
        img_undist = cv2.remap(img_raw, mapx, mapy,
                               interpolation=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        # crop to correct roi
        img_crop = img_undist[y:y + h, x:x + w]

        # Algorithmic line extraction
        bin_mask = algorithmic_extraction(input_image=img_crop, homograph_mtx=H, w=w, h=h)

        # Find lane starts from binary image
        left_line_base, right_line_base = find_line_begining(bin_mask)

        # Cluster mask into two individual lines
        left, right, window_mask = cluster_with_sliding_window(left_line_base, right_line_base, bin_mask)

        # Polyfit to 3rd order polynomial
        left_poly = compute_line_polynom(x=left[0], y=left[1], order=3)
        right_poly = compute_line_polynom(x=right[0], y=right[1], order=3)
        center_poly = (left_poly + right_poly) / 2

        if visualise:
            # create image for showing results
            out_img = np.zeros([h, w, 3])

            # draw binary mask into output image
            out_img[bin_mask == 255] = [255, 255, 255]
            # colour sliding windows
            out_img[window_mask == 255] = [0, 255, 0]
            # colour clustered line pixels
            out_img[h - left[1], w - left[0]] = [255, 0, 0]
            out_img[h - right[1], w - right[0]] = [0, 0, 255]

            # Generate x and y values for plotting lines
            ploty = np.linspace(0, bin_mask.shape[0] - 1, bin_mask.shape[0])
            left_fitx = left_poly[0] * ploty ** 3 + left_poly[1] * ploty ** 2 + left_poly[2] * ploty + left_poly[3]
            right_fitx = right_poly[0] * ploty ** 3 + right_poly[1] * ploty ** 2 + right_poly[2] * ploty + right_poly[3]
            center_fitx = center_poly[0] * ploty ** 3 + center_poly[1] * ploty ** 2 + center_poly[2] * ploty + \
                          center_poly[3]
            # tangent to center polynomial at x = 0
            center_arrowx = center_poly[2] * ploty[0:50] + center_poly[3]

            # Plot lines
            # left line
            cv2.polylines(out_img, [
                np.array(np.concatenate((w - left_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1),
                         np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)
            # right lane
            cv2.polylines(out_img, [
                np.array(np.concatenate((w - right_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1),
                         np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)
            # center lane
            cv2.polylines(out_img, [
                np.array(np.concatenate((w - center_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1),
                         np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)
            # center tangent at x=0
            cv2.polylines(out_img, [np.array(
                np.concatenate(
                    (w - center_arrowx.reshape((-1, 1)), ploty[-len(center_arrowx) - 1:-1].reshape((-1, 1))[::-1]),
                    axis=1), np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
            # image center
            cv2.polylines(out_img, [np.array(np.concatenate(
                ((np.ones(50) * bin_mask.shape[1] // 2).reshape((-1, 1)), ploty[-51:-1].reshape((-1, 1))), axis=1),
                np.int32)], isClosed=False, color=(255, 0, 255), thickness=2)

            cv2.imshow(winname="boxes", mat=out_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    if visualise:
        cv2.destroyAllWindows()
