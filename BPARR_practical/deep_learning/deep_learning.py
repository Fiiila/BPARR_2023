from pathlib import Path
import glob
import cv2
import numpy as np
import time
import tensorflow as tf
from BPARR_practical.preprocessing import generate_undistortion_maps, load_camera_intrinsic, \
    generate_homographic_trans_mtx
from BPARR_practical.postprocessing import find_line_begining, cluster_with_sliding_window, compute_line_polynom

def initialize_nn(model_dir, model_name, img_size, warmup=1):
    # Model parameters


    # initialize GPU and load model
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    # Load model
    model = tf.keras.models.load_model(model_dir + model_name)

    # Model warmup
    if warmup > 0:
        for i in range(warmup):
            _ = model.__call__(np.empty(shape=(1, img_size[0], img_size[1], 3), dtype=np.float32))

    return model


def segmentation_extraction(input_image, model, img_size, homograph_mtx):
    h, w = input_image.shape[:2]
    # use U-Net for extracting lanes
    tmp = np.expand_dims(cv2.resize(input_image, img_size), 0)
    result = model.__call__(tmp)[0]  # predict
    result = tf.identity(result)  # move tensor from GPU to CPU
    result = result.numpy()  # convert predicted tensor to numpy

    # Binary thresholding
    result = result > 0.5
    result = result * 255
    result = result.astype(np.uint8)

    # resize NN output to match with captured image
    resized = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    # warp perspective to bird view
    filter_im_bin = cv2.warpPerspective(resized, homograph_mtx, (w, h))

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

    # Load model
    model_dir = "../deep_learning/models/"  # directory where trained models will be saved
    model_name = "lane_detection.h5"  # name of the trained model
    img_size = (256, 256)  # size of images which will be processed with neural network (only width and height)
    model = initialize_nn(model_dir, model_name, img_size, warmup=1)

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
        bin_mask = segmentation_extraction(input_image=img_crop, model=model, img_size=img_size, homograph_mtx=H)

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
