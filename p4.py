"""Pipeline to find lanes on images or movies."""

import numpy as np
import cv2
import glob
import pickle
import os.path
from moviepy.editor import VideoFileClip
from lane import Lane


DEBUG_VARS = {"show_calibration_images": False,
              "show_original_test_images": False,
              "show_perspective_transform_images": False,
              "show_perspective_transform_reverse_images": False,
              "show_color_transform_images": False,
              "show_processed_test_images": False,
              "show_undistorted_images": False,
              "show_poly_test_images": False}
PROCESS_IMAGES = True
PROCESS_MOVIES = True
saved_camera_calibration_path = './camera_cal/saved_camera_calibration.p'

src_points = np.float32([[240, 690],
                         [1070, 690],
                         [577, 460],
                         [706, 460]])

dst_points = np.float32([[200, 720],
                         [1110, 720],
                         [200, 25],
                         [1110, 25]])

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/880  # meters per pixel in x dimension


def do_camera_calibration():
    """Calculate calibration parameters for all calibration images."""
    # If calibration is saved just load it
    if os.path.isfile(saved_camera_calibration_path):
        return pickle.load(open(saved_camera_calibration_path, "rb"))

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    for image_file in glob.glob('./camera_cal/calibration*.jpg'):
        image = cv2.imread(image_file)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, (9, 6), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if DEBUG_VARS['show_calibration_images']:
                image = cv2.drawChessboardCorners(image, (9, 6), corners, ret)
                cv2.imshow('img', image)
                cv2.waitKey(500)
    camera_calibration = cv2.calibrateCamera(objpoints, imgpoints,
                                             image.shape[1::-1], None, None)
    camera_calibration = {'ret': camera_calibration[0],
                          'mtx': camera_calibration[1],
                          'dist': camera_calibration[2],
                          'rvecs': camera_calibration[3],
                          'tvecs': camera_calibration[4]}
    pickle.dump(camera_calibration, open(saved_camera_calibration_path, "wb"))
    return camera_calibration


def distortion_correction(image):
    """Use the calibration data to undistort the image."""
    image = cv2.undistort(image,
                          camera_calibration['mtx'],
                          camera_calibration['dist'],
                          None,
                          camera_calibration['mtx'])
    if DEBUG_VARS['show_undistorted_images']:
        cv2.imshow('undistorted_image', image)
        cv2.waitKey(1)
    return image


def perspective_transform(image):
    """Transform image to birds-eye view."""
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    image = cv2.warpPerspective(image,
                                M,
                                image.shape[1::-1],
                                flags=cv2.INTER_LINEAR)
    if DEBUG_VARS['show_perspective_transform_images']:
        cv2.imshow('perspective_transform_image', image*255)
        cv2.waitKey(1)
    return image


def perspective_transform_reverse(image):
    """Transfor image back from birds-eye view to normal view."""
    M = cv2.getPerspectiveTransform(dst_points, src_points)
    image = cv2.warpPerspective(image,
                                M,
                                image.shape[1::-1],
                                flags=cv2.INTER_LINEAR)
    if DEBUG_VARS['show_perspective_transform_reverse_images']:
        cv2.imshow('perspective_transform_reverse_image', image*255)
        cv2.waitKey(1)
    return image


def process_image(image_org):
    """Pipeline to find an draw lanes."""
    global line_left, line_right

    image_org = cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR)
    if DEBUG_VARS['show_original_test_images']:
        cv2.imshow('original_test_image', image_org)
        cv2.waitKey(1)
    image_undist = distortion_correction(image_org)
    image_binary = color_transform(image_undist)
    binary_warped = perspective_transform(image_binary)
    line_left, line_right, color_warped = fit_polynomial(binary_warped,
                                                         line_left,
                                                         line_right)

    image = draw_on_road(image_undist, binary_warped, line_left, line_right)
    image = combine_images(image,
                           image1=image_binary,
                           image2=binary_warped,
                           image3=color_warped)
    if DEBUG_VARS['show_processed_test_images']:
        cv2.imshow('processed_test_image', image)
        cv2.waitKey(0)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return image


def combine_images(image, image1=None, image2=None, image3=None):
    """Append optional images below original input image."""
    small_shape = cv2.resize(image, (0, 0), fx=1/3, fy=1/3).shape
    if image1 is None:
        image1 = np.zeros(small_shape)
    else:
        image1 = cv2.resize(image1, small_shape[1::-1])
        if len(image1.shape) < 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)*255
    if image2 is None:
        image2 = np.zeros(small_shape)
    else:
        image2 = cv2.resize(image2, small_shape[1::-1])
        if len(image2.shape) < 3:
            image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)*255
    if image3 is None:
        image3 = np.zeros(small_shape)
    else:
        image3 = cv2.resize(image3, small_shape[1::-1])
        if len(image3.shape) < 3:
            image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2RGB)*255

    image_above = np.concatenate((image1, image2), axis=1)
    image_below = np.concatenate((image_above, image3), axis=1)
    image_below = image_below[:, :1280, :]
    image = np.concatenate((image, image_below), axis=0)
    return cv2.resize(image, (0, 0), fx=0.9, fy=0.9).astype(np.uint8)


def color_transform(image):
    """Calculate binary image with lanes."""
    lanes_hls = get_lanes_hls(image)
    lanes_hsv = get_lanes_hsv(image)

    binaryCom = lanes_hls + lanes_hsv
    binaryCom[binaryCom < 1] = 0
    binaryCom[binaryCom > 0] = 1
    if DEBUG_VARS['show_color_transform_images']:
        cv2.imshow('color_transform_image', binaryCom*255)
        cv2.waitKey(1)
    return binaryCom


def get_lanes_hls(image):
    """Calculate binary image with lanes from hls color space."""
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    binaryH = np.zeros_like(H)
    binaryH[(H >= 16) & (H <= 24)] = 1
    binaryS = np.zeros_like(H)
    binaryS[(S > 170) & (S <= 255)] = 1
    binarySH = np.zeros_like(H)
    binarySH[(S > 70) & (S <= 255) & (H >= 12) & (H <= 28)] = 1
    binaryCom = binaryH + binaryS + binarySH
    binaryCom[binaryCom < 2] = 0
    binaryCom[binaryCom > 1] = 1

    sobel_hls = cv2.Sobel(binaryCom, cv2.CV_64F, 1, 0, ksize=9)
    sobel_hls = np.absolute(sobel_hls)
    sobel_hls = np.uint8(255*sobel_hls/np.max(sobel_hls))
    binary_sobel_hls = np.zeros_like(sobel_hls)
    binary_sobel_hls[(sobel_hls >= 20) & (sobel_hls <= 255)] = 1
    return binary_sobel_hls


def get_lanes_hsv(image):
    """Calculate binary image with lanes from hsv color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv,
                             np.array([[[0, 0, 210]]]),
                             np.array([[[255, 90, 255]]]))
    mask_yellow = cv2.inRange(hsv,
                              np.array([[[19, 120, 150]]]),
                              np.array([[[30, 200, 255]]]))
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    sobel_hsv = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=9)
    sobel_hsv = np.absolute(sobel_hsv)
    sobel_hsv = np.uint8(255*sobel_hsv/np.max(sobel_hsv))
    binary_sobel_hsv = np.zeros_like(sobel_hsv)
    binary_sobel_hsv[(sobel_hsv >= 20) & (sobel_hsv <= 255)] = 1
    return binary_sobel_hsv


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=100):
    """Find lane pixels with sliding window."""
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]*2/3):, :],
                       axis=0)

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_steps = [0]
    rightx_steps = [0]

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_steps.append(np.int(np.mean(nonzerox[good_left_inds]))
                               - leftx_current)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            leftx_current = leftx_current + np.int(np.mean(leftx_steps))
            leftx_steps.append(0)
        if len(good_right_inds) > minpix:
            rightx_steps.append(np.int(np.mean(nonzerox[good_right_inds]))
                                - rightx_current)
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            rightx_current = rightx_current + np.int(np.mean(rightx_steps))
            rightx_steps.append(0)

    # Concatenate the arrays of indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def find_lane_pixels_poly(binary_warped, line_left, line_right, margin=100):
    """Find lane pixels with known previous lanes."""
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_fit = line_left.current_fit
    right_fit = line_right.current_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped, line_left, line_right):
    """Find a fitting polynomial for the left and tight lane."""
    warp_zero = np.zeros_like(binary_warped)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if line_left.bad_Frames < 5 or line_right.bad_Frames < 5:
        leftx, lefty, rightx, righty = find_lane_pixels_poly(binary_warped,
                                                             line_left,
                                                             line_right)
        try:
            # Fit a second order polynomial to each
            poly_fit_left = np.polyfit(lefty, leftx, 2)
            poly_fit_right = np.polyfit(righty, rightx, 2)

            # Generate x and y values
            poly_fity = np.linspace(0, binary_warped.shape[0]-1,
                                    binary_warped.shape[0])
            poly_fitx_left = (poly_fit_left[0]*poly_fity**2 +
                              poly_fit_left[1]*poly_fity +
                              poly_fit_left[2])
            poly_fitx_right = (poly_fit_right[0]*poly_fity**2 +
                               poly_fit_right[1]*poly_fity +
                               poly_fit_right[2])
            line_left_chekcs = line_left.checks(poly_fit_left,
                                                poly_fitx_left,
                                                poly_fity)
            line_right_checks = line_right.checks(poly_fit_right,
                                                  poly_fitx_right,
                                                  poly_fity)

            if (line_left_chekcs and line_right_checks and
                checks(poly_fit_left, poly_fit_right,
                       poly_fitx_left, poly_fitx_right,
                       poly_fity)):
                line_left.update(leftx, lefty, binary_warped.shape)
                line_right.update(rightx, righty, binary_warped.shape)
            else:
                line_left.bad_Frame()
                line_right.bad_Frame()
        except TypeError:
            # Avoids an error if `poly_fitx` still none or incorrect
            print('The function failed to fit a line!')
            line_left.bad_Frame()
            line_right.bad_Frame()

    if line_left.bad_Frames > 4 or line_right.bad_Frames > 4:
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        color_warp = out_img
        if len(leftx) > 0 and len(lefty) > 0:
            line_left.update(leftx, lefty, binary_warped.shape)
        if len(rightx) > 0 and len(righty) > 0:
            line_right.update(rightx, righty, binary_warped.shape)

    # Recast the x and y points into usable format for cv2.fillPoly()
    if line_left.detected:
        poly_left_val = np.polyval(line_left.best_fit,
                                   range(0, binary_warped.shape[0]))
        points_left = np.column_stack((poly_left_val,
                                       range(0, binary_warped.shape[0])))

        color_warp[line_left.ally, line_left.allx] = [255, 0, 0]
        cv2.polylines(color_warp, np.int32([points_left]), False,
                      [255, 255, 255], thickness=4)

    if line_right.detected:
        poly_right_val = np.polyval(line_right.best_fit,
                                    range(0, binary_warped.shape[0]))
        points_right = np.column_stack((poly_right_val,
                                       range(0, binary_warped.shape[0])))

        color_warp[line_right.ally, line_right.allx] = [0, 0, 255]
        cv2.polylines(color_warp, np.int32([points_right]), False,
                      [255, 255, 255], thickness=4)

    return line_left, line_right, color_warp


def checks(poly_fit_left, poly_fit_right,
           poly_fitx_left, poly_fitx_right,
           poly_fity):
    """Check if found lanes make sense."""
    # check curvature
    y_eval = np.max(poly_fity)
    curvature_left = (((1 + (2*poly_fit_left[0]*y_eval +
                             poly_fit_left[1])**2)**1.5) /
                      np.absolute(2*poly_fit_left[0]))
    curvature_right = (((1 + (2*poly_fit_right[0]*y_eval +
                              poly_fit_right[1])**2)**1.5) /
                       np.absolute(2*poly_fit_right[0]))

    if np.maximum(np.log10(curvature_left)/np.log10(curvature_right),
                  np.log10(curvature_right)/np.log10(curvature_left)) > 1.4:
        return False

    # check parallel
    dist = poly_fitx_left-poly_fitx_right
    if np.mean(np.abs(dist-np.mean(dist))) > 100:
        return False
    return True


def draw_on_road(image_undist, binary_warped, line_left, line_right):
    """Draw the found lanes back on the road image."""
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp_lanes = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    poly_fity = np.linspace(0, binary_warped.shape[0]-1,
                            binary_warped.shape[0])
    if line_left.bestx is None or line_right.bestx is None:
        return image_undist
    pts_left = np.array([np.transpose(np.vstack([line_left.bestx,
                                                 poly_fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([line_right.bestx,
                                                            poly_fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    if line_left.best_fit is not None:
        poly_left_val = np.polyval(line_left.best_fit,
                                   range(0, binary_warped.shape[0]))
        points_left = np.column_stack((poly_left_val,
                                       range(0, binary_warped.shape[0])))
        cv2.polylines(color_warp_lanes, np.int32([points_left]), False,
                      [0, 0, 255], thickness=50)

    if line_right.best_fit is not None:
        poly_right_val = np.polyval(line_right.best_fit,
                                    range(0, binary_warped.shape[0]))
        points_right = np.column_stack((poly_right_val,
                                       range(0, binary_warped.shape[0])))
        cv2.polylines(color_warp_lanes, np.int32([points_right]), False,
                      [0, 0, 255], thickness=50)

    # Warp the blank back to original image space
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (image_undist.shape[1],
                                   image_undist.shape[0]))
    newwarp_lanes = cv2.warpPerspective(color_warp_lanes, Minv,
                                        (image_undist.shape[1],
                                         image_undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image_undist, 1, newwarp, 0.3, 0)
    result = cv2.add(result, newwarp_lanes)

    curvature = (line_left.best_radius_of_curvature +
                 line_right.best_radius_of_curvature) / 2
    cv2.putText(result,
                'Curvature: {:7.2f}m'.format(curvature),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    offset = (abs(((line_left.line_base_pos +
                   line_right.line_base_pos) / 2) - binary_warped.shape[1]/2)
              * xm_per_pix)
    cv2.putText(result,
                'Offset: {:4.2f}m'.format(offset),
                (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    return result


camera_calibration = do_camera_calibration()

if PROCESS_IMAGES:
    for test_image_name in os.listdir("test_images/"):
        line_left = Lane()
        line_right = Lane()

        image = cv2.imread('./test_images/'+test_image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = process_image(image)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join('./output_images',
                                 test_image_name),
                    processed_image)

if PROCESS_MOVIES:
    for test_movie_name in glob.glob('*.mp4'):
        clip = VideoFileClip(test_movie_name)
        line_left = Lane()
        line_right = Lane()

        new_clip = clip.fl_image(process_image)
        new_clip.write_videofile(os.path.join('./output_videos',
                                              test_movie_name),
                                 audio=False)
