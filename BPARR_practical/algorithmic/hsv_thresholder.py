import cv2
import numpy as np


def callback(x):
    pass


if __name__=="__main__":

    # cap =cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(f"../res/recorded_video_webcam2.avi")
    cv2.namedWindow('image')

    #[70, 117, 110, 255, 0, 255]

    ilowH = 84
    ihighH = 117

    ilowS = 110
    ihighS = 255

    ilowV = 114
    ihighV = 255

    # create trackbars for color change
    cv2.createTrackbar('lowH','image',ilowH,179,callback)
    cv2.createTrackbar('highH','image',ihighH,179,callback)

    cv2.createTrackbar('lowS','image',ilowS,255,callback)
    cv2.createTrackbar('highS','image',ihighS,255,callback)

    cv2.createTrackbar('lowV','image',ilowV,255,callback)
    cv2.createTrackbar('highV','image',ihighV,255,callback)



    while True:
        # grab the frame
        # ret, frame = cap.read()
        frame = cv2.imread("../resources/jetbot_datasets/dataset_1/000015.png")
        kernel = np.ones((10, 10), np.float32) / 100
        img_gray = cv2.filter2D(frame, -1, kernel)

        # get trackbar positions
        ilowH = cv2.getTrackbarPos('lowH', 'image')
        ihighH = cv2.getTrackbarPos('highH', 'image')
        ilowS = cv2.getTrackbarPos('lowS', 'image')
        ihighS = cv2.getTrackbarPos('highS', 'image')
        ilowV = cv2.getTrackbarPos('lowV', 'image')
        ihighV = cv2.getTrackbarPos('highV', 'image')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # show thresholded image
        cv2.imshow('image', frame)
        k = cv2.waitKey(10) & 0xFF # large wait time to remove freezing
        if k == 113 or k == 27:
            print([ilowH, ihighH, ilowS, ihighS, ilowV, ihighV])
            break