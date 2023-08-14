"""Took from OpenCV guide for camera calibration https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
"""
import sys

import numpy as np
import cv2 as cv
import glob

# custom libraries
from pathlib import Path
import shutil
import json
import time


if __name__=="__main__":
    start = time.time()
    # custom parameters specification
    ## path to dataset
    calib_datset_path = Path("../resources/camera_calibration_dataset")
    ## first preprocessing
    first = False
    calib_sort_path = Path("../resources/camera_calibration_dataset")
    
    if first:
        if not calib_sort_path.exists():
            calib_sort_path.mkdir()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    rows = 9  # no of rows of chessboard
    cols = 6  # no of cols of chessboard
    objp = np.zeros((rows * cols, 3), np.float32)  # initialize object points
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(calib_datset_path.joinpath("*.png").__str__())
    print("Start looking for chessboard in images to calibrate camera\n")
    for i,fname in enumerate(images):
        # if i>50:
        #     continue
        sys.stdout.write(f"\rSearching in {i+1} of {len(images)}")
        sys.stdout.flush()
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            if first:
                # Draw and display the corners
                cv.drawChessboardCorners(img, (cols, rows), corners2, ret)
                # copy original and save displayed chessboard
                shutil.copy(src=fname, dst=calib_sort_path)
                tmp = Path(fname[:-4]+"_corners"+ fname[-4:])
                cv.imwrite(calib_sort_path.joinpath(tmp.name).__str__(), img)
                cv.imshow('img', img)
                cv.waitKey(500)
    print("\n")
    cv.destroyAllWindows()
    print(f"time to find chessboards in all images is: {time.time()-start}")
    start = time.time()

    print("\nGenerating matrices")
    # generate calibration parameters
    ## boolean to signalise if calibration run correctly
    ret = None
    ## camera intrinsic matrix
    mtx = None
    ## distortion coefficients
    dist = None
    ## rotation vectors
    rvecs = None
    ## translation vectors
    tvecs = None

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f"time to generate calibration camera matrices: {time.time()-start}")
    start = time.time()

    # refine camera matrix
    ## refined camera matrix
    refined_mtx = None
    ## suggested roi
    roi = None

    img = cv.imread(images[48])
    h, w = img.shape[:2]
    refined_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    ## undistort test
    # undistort
    dst = cv.undistort(img, mtx, dist, None, mtx)
    cv.imwrite('calibresult_mtx.png', dst)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('calibresult_mtx_roi.png', dst)

    # undistort
    dst = cv.undistort(img, mtx, dist, None, refined_mtx)
    cv.imwrite('calibresult_refined_mtx.png', dst)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv.imwrite('calibresult_refined_mtx_roi.png', dst)

    # re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))

    params = {"ret": ret,
              "mean_error": mean_error / len(objpoints),
              "mtx": mtx.tolist(),
              "refined_mtx": refined_mtx.tolist(),
              "roi": list(roi),
              "dist": dist.tolist(),
              "rvecs": [r.tolist() for r in rvecs],
              "tvecs": [t.tolist() for t in tvecs]
              }

    with open("../camera_intrinsic_calibration.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, separators=(',', ':'))
