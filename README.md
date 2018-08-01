# **Advanced Lane Finding Project**

This repository contains my solution for the project "Advanced Lane Finding Project" of the Udacity Self-Driving Car Engineer Nanodegree Program. The python code could be found in [P4](P4.py), the generated images in the folder [output_images](output_images/) and the videos in the folder [output_videos](output_videos/).

The following part of the README contains a writeup which describes how the lane finding is achieved.

## Writeup

[//]: # (Image References)

[original]: ./writeup_images/original.png "Image before distortion correction"
[distortion_correction]: ./writeup_images/distortion_correction.png "Image before and after distortion correction"
[color_transform]: ./writeup_images/color_transform.png "Binary image after color thresholding and edge filtering"
[perspective_transform]: ./writeup_images/perspective_transform.png "Binary image after perspective transform"
[fit_polynomial]: ./writeup_images/fit_polynomial.png "Binary image after sliding window"
[processed_image]: ./writeup_images/processed_image.png "Original image with drawn lanes"


### Goals

The goal of this project is to detect lanes on images or videos of a road filmed from the top of a car. The individual steps are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In the first step the camera calibration was computed. Therefore the given calibration images which contain the chessboards are loaded and and with the openCV function `cv2.findChessboardCorners()` the corners of the chessboards are found.

After all corners on each image were found the function `cv2.calibrateCamera()` was used to obtain the camera matrix and the distortion coefficients which are returned and saved.

The code for this can be found in `P4.py` in the function `do_camera_calibration()` on line 41.

To not calibrate the camera again on each run the camera matrix and distortion coefficients are saved in the file `camera_cal/saved_camera_calibration.p`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After computing the camera matrix and distortion coefficients the images can be easily corrected by applying the function `cv2.undistort()` on the image. This is done in the first step of the pipeline and can be found in the function `distortion_correction()` on line 78.

Here is an example if an image before and after distortion correction:

![Image before distortion correction][original]
![Image after distortion correction][distortion_correction]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To get a binary image which manly consists of lane pixels color thresholding is used in the HSV and HLS color space. After the thresholding a sobel filter is applied to get vertical edges on the images.

The code can be found in the function `color_transform()` which uses the functions `get_lanes_hls()` and `get_lanes_hsv()` to get binary images out of the HSL and HSV color space. These two binary images are then added.

Here's an example of the output for this step:

![Binary image after color thresholding and edge filtering][color_transform]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To transform the binary image in a birds-eye view the openCV function `cv2.warpPerspective()` is used. Therefore four source and destination points are found by hand and then used in the code.  

The following source and destination points are used:

| Source        | Destination   |
|:-------------:|:-------------:|
| 240, 690      | 200, 720      |
| 1070, 690     | 1110, 720     |
| 577, 460      | 200, 25       |
| 706, 460      | 1110, 25      |

With the give test images with straight lanes one can verify that these source and destination points result in straight parallel lines.

The code for this perspective transform can be found in the function `perspective_transform()`.

Here is an example of the results:

![Binary image after perspective transform][perspective_transform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the lane from the binary image I first tried to Identify all pixels with are part of the left or tight lane. To do this the sliding window approach is used. This approach starts at the bottom of the image and works its way of in small windows to identify the lane pixels. This code can be found in the function `find_lane_pixels()`.

If there were already found both lanes in a previous image of the same movie the pixels are just searched for in a margin around the previous lane. In the code this is implemented in the function `find_lane_pixels_poly()`.

In both functions a second order polynomial is fitted to all found pixels to obtain a line.

If there were lanes found before but the new found lanes didn't make sense the old lanes are reused. If this happens for a few following images the sliding window approach is used again.

This and the usage of `find_lane_pixels()` and `find_lane_pixels_poly()` can be found in `fit_polynomial()`.

Here is an image with an image after the sliding window approach:
![Binary image after sliding window][fit_polynomial]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TO calculate the curvature I used this
[formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) together with a conversion from pixel to meter. In the code this can be found in the file `lane.py` in the function `update()` in the lines 97 to 102.

To calculate the offset of the car from the center of the road I measured the distance from the center of the image to the right and left lane. This happens in line 511 in the function `draw_on_road()` in `P4.py`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Drawing the found lanes back on the original image is implemented in the function `draw_on_road()`.
Here is an example of the result on a test image:

![Original image with drawn lanes][processed_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult and error-prone part of the pipeline is to get a good binary image. The colors of the lanes can't be easily detected and Therefore we get too few or too much pixels in the binary image.
The rest of the pipeline works pretty well.
