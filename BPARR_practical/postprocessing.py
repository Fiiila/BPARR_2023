import numpy as np
import cv2


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


def cluster_with_sliding_window(left_line_base, right_line_base, filter_im_bin):
    """Function for line clustering with sliding window method

    - code is used from https://github.com/muddassir235/Advanced-Lane-and-Vehicle-Detection and slightly modified for
    current problem

    :param left_line_base: x position of left line beginning
    :param right_line_base: x position of left line beginning
    :param filter_im_bin: generated mask with found line pixels with value 255 (highest)
    :return: pixels belonging to individual lines
    """

    h, w = filter_im_bin.shape[:2]
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
    # output mask for plotting
    out_img = np.zeros([h, w])
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
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255), 2)
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

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)

    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    # transform pixels
    leftx = w - nonzero_x[left_lane_inds]
    lefty = h - nonzero_y[left_lane_inds]
    rightx = w - nonzero_x[right_lane_inds]
    righty = h - nonzero_y[right_lane_inds]

    return (leftx, lefty), (rightx, righty), out_img


def compute_line_polynom(x, y, order=3):
    line_fit = np.polyfit(y, x, order)
    return line_fit

