import glob
import json
from pathlib import Path
import cv2
import numpy as np
import time
from line_profiler_pycharm import profile
import tensorflow as tf
import pandas as pd


def load_camera_intrinsic(calib_path):
    with open(calib_path, "r", encoding="utf-8") as calib_file:
        json_params = json.load(calib_file)
    calib_params = {"ret": json_params["ret"],
                    "mean_error": json_params["mean_error"],
                    "mtx": np.array(json_params["mtx"]),
                    "refined_mtx": np.array(json_params["refined_mtx"]),
                    "roi": json_params["roi"],
                    "dist": np.array(json_params["dist"][0]),
                    "rvecs": np.array([[r1[0] for r1 in r] for r in json_params["rvecs"]]),
                    "tvecs": np.array([[t1[0] for t1 in t] for t in json_params["tvecs"]])
                    }
    return calib_params

def create_steerable_filters(kernel_size, variance=(2,2), center=(0, 0), amplitude=1):
    # 2D Gaussian
    x = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    y = x.copy()
    x0, y0 = center
    varx, vary = variance
    gauss_kernel = np.zeros((len(y), len(x)))
    gauss_kernel_diff = np.zeros((len(y), len(x)))
    for idy in range(len(y)):
        for idx in range(len(x)):
            gauss_kernel[idy, idx] = amplitude*np.exp(-((np.square(x[idx]-x0)/np.square(2*varx))+(np.square(y[idy]-y0)/np.square(2*vary))))
            gauss_kernel_diff[idy, idx] = -(np.exp(-np.square(x[idx] - x0)/(2*varx) - np.square(y[idy] - y0)/(2*vary))*(2*x[idx] - 2*x0))/(2*varx)

    tmp = np.gradient(gauss_kernel)
    return gauss_kernel_diff

def find_line_begining(img):
    height, width = img.shape[:2]
    # histogram for columns of image
    histogram = np.sum(img[-height // 6:-1, :], axis=0)  # find beginning only in down half
    middle = width // 2
    # peaks = find_peaks(histogram, height = 255*(height//6)//8, distance = 20)
    # instead of peaks
    noOfRanges = 10
    width_range = width // noOfRanges
    prevRight = 0
    left_line_base = 0
    right_line_base = width
    local_max = np.zeros((2, 2), dtype=int)
    temp = 0
    tempdist = 0
    for i in range(0, noOfRanges):
        temp = np.argmax([histogram[prevRight:prevRight + width_range]]) + prevRight
        tempdist = temp - tempdist
        if local_max[0, 1] < histogram[temp] and ((temp - local_max[0, 0]) > width_range) and (
                (temp - local_max[1, 0]) > width_range):
            if local_max[1, 1] < histogram[temp] and ((temp - local_max[0, 0]) > width_range) and (
                    (temp - local_max[1, 0]) > width_range):
                local_max[0, :] = local_max[1, :]
                local_max[1, :] = np.array((temp, histogram[temp]), dtype=int)
            else:
                local_max[0, :] = np.array((temp, histogram[temp]), dtype=int)

        prevRight += width_range

    if local_max[0, 0] < local_max[1, 0]:
        left_line_base = local_max[0, 0]
        right_line_base = local_max[1, 0]
    else:
        left_line_base = local_max[1, 0]
        right_line_base = local_max[0, 0]
    return left_line_base, right_line_base

def generate_homographic_trans_mtx(img, chess_cols=7, chess_rows=5):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    rows = chess_rows  # no of rows of chessboard
    cols = chess_cols  # no of cols of chessboard
    objp = np.zeros((rows * cols, 3), np.float32)  # initialize object points
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(gray, (cols, rows), corners2, ret)
        # cv2.drawMarker(gray, (int(corners2[0][0][0]),int(corners2[0][0][1])), (255), 1, 10, thickness=4)
        # cv2.imshow("test", gray)
        # cv2.waitKey(0)
    else:
        cv2.imshow("test", gray)
        cv2.waitKey(0)
        print("unsuccesfull chessboard searching")
    # find corners of chessboard
    tl = tuple(corners2[-1][0])
    tr = tuple(corners2[-chess_cols][0])
    dl = tuple(corners2[chess_cols-1][0])
    dr = tuple(corners2[0][0])
    h, w = img.shape[:2]
    # tl = [w, h]
    # tr = [0, h]
    # dl = [w, 0]
    # dr = [0, 0]

    # corners3 = np.copy(corners2)
    # np.random.shuffle(corners3)
    # for corner in corners3:
    #     if corner[0][0] < tl[0] or corner[0][1] < tl[1]:
    #         # top left
    #         tl = corner[0]
    #     if corner[0][0] > tr[0] or corner[0][1] < tr[1]:
    #         # top right
    #         tr = corner[0]
    #     if corner[0][0] < dl[0] and corner[0][1] > dl[1]:
    #         # down left
    #         dl = corner[0]
    #     if corner[0][0] > dr[0] and corner[0][1] > dr[1]:
    #         # down right
    #         dr = corner[0]

    # get perspective matrix
    chess_rows += 1
    chess_cols += 1
    a = np.sqrt((tr[0]-tl[0])**2+(tr[1]-tl[1])**2)/chess_cols
    # print(f"rectangle edge\n{a}px\n{31}mm\n1 px = {31/a:.4f} mm")
    H = cv2.getPerspectiveTransform(src=np.float32([tl, tr, dr, dl]), dst=np.float32([(tl[0], 0+h/chess_rows), (tr[0], 0+h/chess_rows), (tr[0], h-h/chess_rows), (tl[0], h-h/chess_rows)]))
    # H = cv2.getPerspectiveTransform(src=np.float32([tl, tr, dr, dl]), dst=np.float32(
    #     [(tl[0], 0), (tr[0], 0), (tr[0], a*chess_rows),
    #      (tl[0], a*chess_rows)]))

    # cv2.drawMarker(gray, (int(tl[0]), int(tl[1])), (255), 1, 10, thickness=4)
    # cv2.imshow("tl", gray)
    # cv2.waitKey(0)
    # cv2.drawMarker(gray, (int(tr[0]), int(tr[1])), (255), 2, 10, thickness=4)
    # cv2.imshow("tr", gray)
    # cv2.waitKey(0)
    # cv2.drawMarker(gray, (int(dl[0]), int(dl[1])), (255), 3, 10, thickness=4)
    # cv2.imshow("dl", gray)
    # cv2.waitKey(0)
    # cv2.drawMarker(gray, (int(dr[0]), int(dr[1])), (255), 4, 10, thickness=4)
    # cv2.imshow("dr", gray)
    # cv2.waitKey(0)
    return H



@profile
def main():
    # parameters for logging
    log_path = "logs/logging_nn_validation01.csv"
    preprocessing = []
    lane_extraction = []
    postprocessing = []
    p1_l = []  # polynom parameter for order 3
    p1_r = []
    p2_l = []  # polynom parameter for order 2
    p2_r = []
    p3_l = []  # polynom parameter for order 1
    p3_r = []
    p4_l = []  # polynom parameter for order 0
    p4_r = []
    k = []  # parameter of linear tangent line at x=0
    q = []  # parameter of linear tangent line at x=0
    dx = []  # distance of center line from center of image

    # switch between methods
    use_u_net = True
    warmup = 1

    # load created camera calibration parameters
    calib_path = Path("../camera_intrinsic_calibration.json")
    params = load_camera_intrinsic(calib_path=calib_path)

    # transfer parameters from dictionary into parameters for undistortion
    mtx = params["mtx"]
    refined_mtx = params["refined_mtx"]
    dist = params["dist"]
    roi = params["roi"]
    ## specify roi cutting --> need to be adjusted if source images are different from calibration dataset
    x, y, w, h = roi

    # generate homographic transformation matrix
    img_undist = cv2.undistort(src=cv2.imread("../resources/jetbot_datasets/dataset_4/000000.png"), cameraMatrix=mtx, distCoeffs=dist, dst=None, newCameraMatrix=refined_mtx)
    img_crop = img_undist[y:y + h, x:x + w]
    homograph_mtx = generate_homographic_trans_mtx(img=img_crop)

    # cycle through images/video frames
    video_path = Path("../resources/jetbot_datasets/dataset_10")#("../resources/jetbot_datasets/dataset_10")
    images = glob.glob(video_path.joinpath("*.png").__str__())
    h_im, w_im = cv2.imread(images[0]).shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, refined_mtx, (w_im, h_im), cv2.CV_32FC1)
    if use_u_net:
        # Model parameters
        model_dir = "../deep_learning/models/"  # directory where trained models will be saved
        model_name = "lane_detection.h5"  # name of the trained model
        img_size = (256, 256)  # size of images which will be processed with neural network (only width and height)


        # initialize GPU and load model
        start = time.time()
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
        print(f"Model loading and GPU setup time: {time.time()-start:.4f} s.")

        # Model warmup
        if warmup > 0:
            start = time.time()
            for i in range(warmup):
                _ = model.__call__(np.empty(shape=(1, 256, 256, 3), dtype=np.float32))
            print(f"Model warmup time: {time.time() - start:.4f} s.")

    else:
        # create steerable filter
        fltr_rl = create_steerable_filters(kernel_size=10, variance=(8, 1), center=(0, 0))
        fltr_lr = fltr_rl[:, ::-1]
        pass
    for image_number, image in enumerate(images):
        if image_number % 5 != 0:
            continue
        # load raw image
        img_raw = cv2.imread(image)

        # undistort
        start = time.time()
        # alternative but slow way to undistort sequential images
        # img_undist = cv2.undistort(src=img_raw, cameraMatrix=mtx, distCoeffs=dist, dst=None, newCameraMatrix=refined_mtx)
        img_undist = cv2.remap(img_raw, map1, map2,
                              interpolation=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))

        # crop to correct roi
        img_crop = img_undist[y:y + h, x:x + w]

        preprocessing.append(time.time()-start)

        if use_u_net:
            # use U-Net for extracting lanes
            start = time.time()
            tmp = np.expand_dims(cv2.resize(img_crop, img_size), 0)
            result = model.__call__(tmp)[0]  # predict
            result = tf.identity(result)  # move tensor from GPU to CPU
            result = result.numpy()  # convert predicted tensor to numpy

            # Binary thresholding
            result = result > 0.5
            result = result * 255
            result = result.astype(np.uint8)

            # resize NN output to match with captured image
            resized = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

            # Logging
            lane_extraction.append(time.time() - start)
            print(f"Lane extraction - algorithmic time: {lane_extraction[-1]} s")

            # warp perspective
            filter_im_bin = cv2.warpPerspective(resized, homograph_mtx, (w, h))
        else:
            # warp perspective
            img_warped = cv2.warpPerspective(img_crop, homograph_mtx, (w, h))

            # convert image to grayscale for edge detection
            start = time.time()
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
            filter_im_bin_lr = ((filter_im1 > thr_bin)*255).astype(dtype="int")
            filter_im_bin = (((filter_im_bin_rl+filter_im_bin_lr) >= 255)*255).astype(dtype="uint8")

            # mask out unwanted edges
            cutout_width = 30  # number of px, that will be extracted from middle to move closer...needs to be odd
            final_mask = np.zeros((h, w), dtype="uint8")
            mask = ((img_gray > 0)*1).astype(dtype="uint8")
            final_mask[:, cutout_width // 2:w // 2] += mask[:, 0:w // 2 - cutout_width // 2]
            final_mask[:, w // 2:w - cutout_width // 2] += mask[:, w // 2 + cutout_width // 2:w]
            filter_im_bin *= final_mask

            # Logging
            lane_extraction.append(time.time() - start)
            print(f"Lane extraction - algorithmic time: {lane_extraction[-1]} s")

        # # histogram for columns of image
        # histogram = np.sum(filter_im_bin[-h // 6:-1, :], axis=0)  # find beginning only in down half
        # middle = w // 2
        # # peaks = find_peaks(histogram, height = 255*(height//6)//8, distance = 20)
        # # instead of peaks
        # noOfRanges = 10
        # width_range = w // noOfRanges
        # prevRight = 0
        # left_line_base = 0
        # right_line_base = w
        # local_max = np.zeros((2, 2), dtype=np.int32)
        # temp = 0
        # tempdist = 0
        # for i in range(0, noOfRanges):
        #     temp = np.argmax([histogram[prevRight:prevRight + width_range]]) + prevRight
        #     tempdist = temp - tempdist
        #     if local_max[0, 1] < histogram[temp] and ((temp - local_max[0, 0]) > width_range) and (
        #             (temp - local_max[1, 0]) > width_range):
        #         if local_max[1, 1] < histogram[temp] and ((temp - local_max[0, 0]) > width_range) and (
        #                 (temp - local_max[1, 0]) > width_range):
        #             local_max[0, :] = local_max[1, :]
        #             local_max[1, :] = np.array((temp, histogram[temp]), dtype=np.int32)
        #         else:
        #             local_max[0, :] = np.array((temp, histogram[temp]), dtype=np.int32)
        #
        #     prevRight += width_range
        #
        # if local_max[0, 0] < local_max[1, 0]:
        #     left_line_base = local_max[0, 0]
        #     right_line_base = local_max[1, 0]
        # else:
        #     left_line_base = local_max[1, 0]
        #     right_line_base = local_max[0, 0]
        # '''
        # if(len(peaks[0])>=2):
        #     print(peaks)
        #     left_line_base = peaks[0][0]#np.argmax(histogram[:middle])
        #     right_line_base = peaks[0][1]#np.argmax(histogram[middle:]) + middle
        #     '''

        # Find lane starts from binary image
        start = time.time()
        left_line_base, right_line_base = find_line_begining(filter_im_bin)

        # Cluster lanes with sliding windows method
        # define number of line windows
        no_of_windows = 10
        # count individual window height
        window_height = int(h / no_of_windows)
        # find nonzero elements
        nonzero = filter_im_bin.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        # current position in windows
        left_line_current = int(left_line_base)
        right_line_current = int(right_line_base)
        # width od windows
        window_width = 40
        margin = window_width
        # Set minimum number of pixels found to recenter window
        minpix = 10
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        out_img = np.zeros([h, w, 3])
        out_img[:, :, 0] = np.copy(filter_im_bin)
        out_img[:, :, 1] = np.copy(filter_im_bin)
        out_img[:, :, 2] = np.copy(filter_im_bin)
        # stopper to stop looking for next boxes if there are any
        stopper_left = False
        stopper_right = False
        # Step through the windows one by one
        for window in range(no_of_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = filter_im_bin.shape[0] - (window + 1) * window_height
            win_y_high = filter_im_bin.shape[0] - window * window_height
            if not stopper_left:
                win_xleft_low = int(left_line_current - margin)
                win_xleft_high = int(left_line_current + margin)
            win_xright_low = int(right_line_current - margin)
            win_xright_high = int(right_line_current + margin)
            # Draw the windows on the visualization image
            if not stopper_left:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            if not stopper_left:
                good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low) & (
                        nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low) & (
                        nonzero_x < win_xright_high)).nonzero()[0]

            if (len(good_left_inds) < minpix) and (window > 0):
                stopper_left = True
            else:
                stopper_left = False

            # Append these indices to the lists
            if not stopper_left:
                left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                # tmp = int(np.mean(nonzero_x[good_left_inds]))
                # left_line_current = tmp + (tmp - (win_xleft_low+window_width//2))
                left_line_current = int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                # tmp = int(np.mean(nonzero_x[good_right_inds]))
                # right_line_current = tmp + (tmp - (win_xright_low+window_width//2))
                right_line_current = int(np.mean(nonzero_x[good_right_inds]))

        # POLYFIT
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)

        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        # transform pixels
        leftx = w - nonzero_x[left_lane_inds]
        lefty = h - nonzero_y[left_lane_inds]
        rightx = w - nonzero_x[right_lane_inds]
        righty = h - nonzero_y[right_lane_inds]
        if len(leftx) > 1 and len(rightx) > 1:
            # Fit a third order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 3)

            right_fit = np.polyfit(righty, rightx, 3)

            center_fit = np.array(
                [(left_fit[0] + right_fit[0]) / 2, (left_fit[1] + right_fit[1]) / 2, (left_fit[2] + right_fit[2]) / 2,
                 (left_fit[3] + right_fit[3]) / 2])

            difference_px = center_fit[3] - filter_im_bin.shape[1] // 2

            # Logging
            postprocessing.append(time.time()-start)
            p1_l.append(left_fit[0])
            p1_r.append(right_fit[0])
            p2_l.append(left_fit[1])
            p2_r.append(right_fit[1])
            p3_l.append(left_fit[2])
            p3_r.append(right_fit[2])
            p4_l.append(left_fit[3])
            p4_r.append(right_fit[3])
            k.append(center_fit[2])
            q.append(center_fit[3])
            dx.append(difference_px)

            # Generate x and y values for plotting
            ploty = np.linspace(0, filter_im_bin.shape[0] - 1, filter_im_bin.shape[0])
            left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
            right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
            center_fitx = center_fit[0] * ploty ** 3 + center_fit[1] * ploty ** 2 + center_fit[2] * ploty + center_fit[
                3]

            # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            # center_fitx = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]
            # center_arrow_fit = np.polyfit(ploty[-3:-1], center_fitx[-3:-1], 1)

            # center_arrowx = center_arrow_fit[1] * ploty[-50:-1] + center_arrow_fit[0]
            center_arrowx = center_fit[2] * ploty[0:50] + center_fit[3]


            out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

            # left line
            cv2.polylines(out_img, [
                np.array(np.concatenate((w-left_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1), np.int32)],
                         isClosed=False, color=(0, 255, 255), thickness=2)
            # right lane
            cv2.polylines(out_img, [
                np.array(np.concatenate((w-right_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1), np.int32)],
                         isClosed=False, color=(0, 255, 255), thickness=2)
            # center lane
            cv2.polylines(out_img, [
                np.array(np.concatenate((w-center_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1), np.int32)],
                         isClosed=False, color=(0, 255, 255), thickness=2)
            #
            cv2.polylines(out_img, [np.array(
                np.concatenate((w-center_arrowx.reshape((-1, 1)), ploty[-len(center_arrowx) - 1:-1].reshape((-1, 1))[::-1]),
                               axis=1), np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.polylines(out_img, [np.array(np.concatenate(
                ((np.ones(50) * filter_im_bin.shape[1] // 2).reshape((-1, 1)), ploty[-51:-1].reshape((-1, 1))), axis=1),
                                            np.int32)], isClosed=False, color=(255, 0, 255), thickness=2)
        else:
            # Logging
            postprocessing.append(time.time()-start)
            p1_l.append(np.nan)
            p1_r.append(np.nan)
            p2_l.append(np.nan)
            p2_r.append(np.nan)
            p3_l.append(np.nan)
            p3_r.append(np.nan)
            p4_l.append(np.nan)
            p4_r.append(np.nan)
            k.append(np.nan)
            q.append(np.nan)
            dx.append(np.nan)

        # display output
        if not use_u_net:
            cv2.imshow(winname="gray blurred features image", mat=img_gray)
            cv2.imshow(winname="warped image", mat=img_warped)
        # cv2.imwrite("./pokus.png", out_img)
        cv2.imshow(winname="extracted features image", mat=filter_im_bin)
        cv2.imshow(winname="cropped image", mat=img_crop)
        cv2.imshow(winname="boxes", mat=out_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    log = {"preprocessing": preprocessing,
           "lane_extraction": lane_extraction,
           "postprocessing": postprocessing,
           "p1_l": p1_l,
           "p2_l": p2_l,
           "p3_l": p3_l,
           "p4_l": p4_l,
           "p1_r": p1_r,
           "p2_r": p2_r,
           "p3_r": p3_r,
           "p4_r": p4_r,
           "k": k,
           "q": q,
           "dx": dx}
    df = pd.DataFrame(log)
    df.to_csv(log_path)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()