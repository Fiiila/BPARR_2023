from pathlib import Path
import glob
import cv2
from BPARR_practical.comparison.comparison import load_camera_intrinsic



if __name__=="__main__":
    source = Path("../resources/jetbot_datasets/dataset_10")
    destination = Path("./datasets/dataset_10_undist")
    destination.mkdir(parents=True, exist_ok=True)
    images = glob.glob(f"{source.__str__()}/*.png")

    # load undistortion parameters
    params = load_camera_intrinsic("../camera_intrinsic_calibration.json")
    # transfer parameters from dictionary into parameters for undistortion
    mtx = params["mtx"]
    refined_mtx = params["refined_mtx"]
    dist = params["dist"]
    roi = params["roi"]
    ## specify roi cutting --> need to be adjusted if source images are different from calibration dataset
    x, y, w, h = roi
    for i, image in enumerate(images):
        if not i % 5 == 0:
            continue
        im = cv2.imread(image)
        if i == 0:
            map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, refined_mtx, (im.shape[1], im.shape[0]), cv2.CV_32FC1)
        im_undist = cv2.remap(im, map1, map2,
                              interpolation=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        im_crop = im_undist[y:y + h, x:x + w]
        cv2.imwrite(f"{destination.__str__()}/{i:03}.png", im_crop)
        cv2.rectangle(im_undist, (x, y), (x + w, y + h), thickness=2, color=(255, 0, 255))
        # cv2.imwrite(f"{destination.__str__()}/{i:03}.png", im_undist)
        cv2.imshow("undist", im_undist)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()