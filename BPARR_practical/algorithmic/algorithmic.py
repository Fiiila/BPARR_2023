import glob
import json
from pathlib import Path
import cv2
import numpy as np


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
    histogram = np.sum(filter_im_bin[-height // 6:-1, :], axis=0)  # find beginning only in down half
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
        cv2.imshow("test", gray)
        cv2.waitKey(0)
    else:
        cv2.imshow("test", gray)
        cv2.waitKey(0)
        print("unsuccesfull chessboard searching")
    # find corners of chessboard
    tl = tuple(corners2[-1][0])
    tr = tuple(corners2[-chess_cols][0])
    dl = tuple(corners2[chess_cols-1][0])
    dr = tuple(corners2[0][0])
    # h, w = img.shape[:2]
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
    chess_rows+=1
    chess_cols+=1
    H = cv2.getPerspectiveTransform(src=np.float32([tl, tr, dr, dl]), dst=np.float32([(0+w/chess_cols, 0+h/chess_rows), (w-w/chess_cols, 0+h/chess_rows), (w-w/chess_cols, h-h/chess_rows), (0+w/chess_cols, h-h/chess_rows)]))

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



if __name__ == "__main__":
    # load created camera calibration parameters
    calib_path = Path("./camera_intrinsic_calibration.json")
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
    video_path = Path("../resources/jetbot_datasets/dataset_1")
    images = glob.glob(video_path.joinpath("*.png").__str__())
    for image in images:
        # load raw image
        img_raw = cv2.imread(image)
        # undistort
        img_undist = cv2.undistort(src=img_raw, cameraMatrix=mtx, distCoeffs=dist, dst=None, newCameraMatrix=refined_mtx)
        # crop to correct roi
        img_crop = img_undist[y:y + h, x:x + w]

        # warp perspective
        img_warped = cv2.warpPerspective(img_crop, homograph_mtx, (w, h))

        # convert image to grayscale for edge detection
        img_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

        # create steerable filter
        fltr = create_steerable_filters(kernel_size=10, variance=(8, 1), center=(0, 0))
        # gaussian blurring
        kernel = np.ones((10, 10), np.float32) / 100
        img_gray = cv2.filter2D(img_gray, -1, kernel)

        # feature extraction
        filter_im = cv2.filter2D(src=img_gray, ddepth=-1, kernel=fltr)
        # binary thresholding
        thr_bin, filter_im_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#(filter_im >= thr_bin)*255.0



        # display output
        cv2.imshow(winname="gray blurred features image", mat=img_gray)
        cv2.imshow(winname="extracted features image", mat=filter_im_bin)
        cv2.imshow(winname="cropped image", mat=img_crop)
        cv2.imshow(winname="warped image", mat=img_warped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()