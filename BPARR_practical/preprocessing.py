import cv2
import json
import numpy as np


def generate_undistortion_maps(mtx, dist, refined_mtx, w_im, h_im):
    """Function to generate matrices for image undistortion

    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param refined_mtx: new refined camera matrix
    :param w_im: width of image before undistortion
    :param h_im: height of image before undistortion
    :return: maps for undistorting source image for x and y axis
    """
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, refined_mtx, (w_im, h_im), cv2.CV_32FC1)
    return mapx, mapy


def load_camera_intrinsic(calib_path):
    # read json into dictionary
    with open(calib_path, "r", encoding="utf-8") as calib_file:
        json_params = json.load(calib_file)

    # transfer value types back to original
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


def generate_homographic_trans_mtx(img, chess_cols=7, chess_rows=5):
    show_progress = False
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
        if show_progress:
            cv2.drawChessboardCorners(gray, (cols, rows), corners2, ret)
            cv2.drawMarker(gray, (int(corners2[0][0][0]),int(corners2[0][0][1])), (255), 1, 10, thickness=4)
            cv2.imshow("test", gray)
            cv2.waitKey(0)
    else:
        print("unsuccesfull chessboard searching")
        if show_progress:
            cv2.imshow("test", gray)
            cv2.waitKey(0)

    # find corners of chessboard
    tl = tuple(corners2[-1][0])
    tr = tuple(corners2[-chess_cols][0])
    dl = tuple(corners2[chess_cols-1][0])
    dr = tuple(corners2[0][0])
    h, w = img.shape[:2]

    # get perspective matrix
    chess_rows += 1
    chess_cols += 1

    H = cv2.getPerspectiveTransform(src=np.float32([tl, tr, dr, dl]), dst=np.float32([(tl[0], 0+h/chess_rows), (tr[0], 0+h/chess_rows), (tr[0], h-h/chess_rows), (tl[0], h-h/chess_rows)]))
    # alternative to method above
    # a = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2) / chess_cols
    # H = cv2.getPerspectiveTransform(src=np.float32([tl, tr, dr, dl]), dst=np.float32(
    #     [(tl[0], 0), (tr[0], 0), (tr[0], a*chess_rows),
    #      (tl[0], a*chess_rows)]))
    return H
