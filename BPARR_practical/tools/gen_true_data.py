from pathlib import Path
import cv2
from BPARR_practical.comparison.comparison import load_camera_intrinsic, generate_homographic_trans_mtx
import glob
import numpy as np
import pandas as pd

def main():
    # parameters for logging
    log_path = "../comparison/logs/logging_validation_true.csv"
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
    use_u_net = False
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
    img_undist = cv2.undistort(src=cv2.imread("../resources/jetbot_datasets/dataset_4/000000.png"), cameraMatrix=mtx,
                               distCoeffs=dist, dst=None, newCameraMatrix=refined_mtx)
    img_crop = img_undist[y:y + h, x:x + w]
    homograph_mtx = generate_homographic_trans_mtx(img=img_crop)

    # cycle through images/video frames
    video_path = Path("../deep_learning/datasets/dataset_validation/masks")
    images = glob.glob(video_path.joinpath("*.png").__str__())
    for image in images:
        # load raw image
        img_raw = cv2.imread(image)

        # warp perspective
        img_raw = cv2.warpPerspective(img_raw, homograph_mtx, (w, h))

        # extract right line pixels
        right_line_px = (img_raw == 3).nonzero()

        # extract left line pixels
        left_line_px = (img_raw == 2).nonzero()

        # fit to polynom
        lefty = h - left_line_px[0]
        leftx = w - left_line_px[1]
        left_fit = np.polyfit(lefty, leftx, 3)

        righty = h - right_line_px[0]
        rightx = w - right_line_px[1]
        right_fit = np.polyfit(righty, rightx, 3)

        p1_l.append(left_fit[0])
        p2_l.append(left_fit[1])
        p3_l.append(left_fit[2])
        p4_l.append(left_fit[3])
        p1_r.append(right_fit[0])
        p2_r.append(right_fit[1])
        p3_r.append(right_fit[2])
        p4_r.append(right_fit[3])
        k.append((left_fit[2] + right_fit[2]) / 2)
        q.append((left_fit[3] + right_fit[3]) / 2)
        dx.append(q[-1] - w // 2)

        filter_im_bin = (img_raw>0)*255

        # Generate x and y values for plotting
        ploty = np.linspace(0, filter_im_bin.shape[0] - 1, filter_im_bin.shape[0])
        left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
        right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]

        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # center_fitx = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]
        # center_arrow_fit = np.polyfit(ploty[-3:-1], center_fitx[-3:-1], 1)

        # center_arrowx = center_arrow_fit[1] * ploty[-50:-1] + center_arrow_fit[0]
        center_arrowx = k[-1] * ploty[0:50] + q[-1]

        filter_im_bin[:, :] = [0, 0, 0]
        filter_im_bin[left_line_px[:2]] = [255, 0, 0]
        filter_im_bin[right_line_px[:2]] = [0, 0, 255]

        # left line
        cv2.polylines(filter_im_bin, [
            np.array(np.concatenate((w - left_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1), np.int32)],
                      isClosed=False, color=(0, 255, 255), thickness=2)
        # right lane
        cv2.polylines(filter_im_bin, [
            np.array(np.concatenate((w - right_fitx.reshape((-1, 1)), ploty.reshape((-1, 1))[::-1]), axis=1),
                     np.int32)],
                      isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.imshow(winname="result", mat=filter_im_bin.astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    log = {"p1_l": p1_l,
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
    print("HOTOVO")
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()