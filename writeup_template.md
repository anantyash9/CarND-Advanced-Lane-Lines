**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cell of the IPython notebook Advanced_pipeline.ipynb. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The camera calibration and distortion coefficients are computed in code cell 2 & 3 of Advanced_pipeline.ipynb using the `cv2.calibrateCamera()` function. These cofficients are calculated only once and are then stored in global variables. These cofficients are used to correct distortion in image passed to the pipeline using `cv2.undistort()`. This is done in code cell 5.


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in code cell 5 of Advanced_pipeline.ipynb).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

To get this output perform the following :

* Convert to HSV space and extract V channel from HSV space (Hue, Saturation, Value). This is good at picking the yellow line in varying light conditions.

* Use sobel x on the extracted value channel. This is better at picking lines at the very end of region of interest.

* Use the blue channel in RGB to pick the dashed white line. I can use a higher threshold value as the yellow has no blue component.

* Create a blank binary image to combine the v channel, b channel and sobel x layers with individual threhold value.

* Tune the threshhold values until the lane lines in all parts of the video are present in the binary image. 

* Use A suitable mask to block out everything that is outside the region of interest

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the function called `pre_prespective()` which is in the cell 5 of Advanced_pipeline.ipynb. This function also includes all other pre-processing required befor prespective transformation. This function stores prespective transformation Matrix only once and stores it in a global variable. It returns the transformed image and the Invers transformation matrix as output.
I chose the hardcode the source and destination points in the following manner:

```python
        src = np.array([[210,700],[1070,700],[690,450],[590,450]],np.float32)
        dsts = np.array([[352,720],[928,720],[928,20],[352,20]],np.float32)
```

I used MS Paint to get the source points from the edges of the road. I calculated the destination points based on the fact that the image i used to get the source points had straight road so the destination points would form a rectangle. 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This is done in the code cells 6 and 7 of Advanced_pipeline.ipynb. The functions `get_fits()` and `get_fit_contineous()` are used to identified lane-line pixels and fit their positions with a polynomial.

`get_fits()` is used when we have no idea where the lane line was on the previous frame. It uses histogram for the bottom half of the image to get the position of the right and left lane lines. From there sliding windows are used to get indices of points on right and left lane. These windows have a width and the center of the windows change when a certain number of pixeles are dectected. Once the pixel positions of points on right and left lane are identified fit a second order polynomial to each in the following manner.

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

`get_fit_contineous()` is used when the probable position of the lane line is already known. After the first frame is processed this function is called and the second order polynomial from the last frame are passed to it. These polynomials are used to do a highly targeted search for points on left and right lanes. The size of the region where new points are searched is controlled by tweeking the margine. The search for new points is done in the following manner.

```python
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))  
```
The points are then used to get the polynomial for the lane lines.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the code cell 8 of Advanced_pipeline.ipynb which has the `radii()` function. This function calculates the radius of curvature and deviation from the center of the lane and also displays them on the image.

The function calculates the radius of curvature of the lane lines independently and then averages to gives the combined radius of curvature. 
For each line the radius is calculated as :

R 	 = ((1+(2Ay+B)^2 )^(3/2) / )∣2A∣	

A scaling factor is used for x and y to convert the radius from pixles to meters in the following manner.

```python
    ym_per_pix = 30/700 # meters per pixel in y dimension
    xm_per_pix = 3.7/576 # meters per pixel in x dimension
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in code cell 9 of Advanced_pipeline.ipynb in the function `draw_road()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)
My Pipeline uses the average value from the past 11 frames for x and y points for the lanes. This ensures that one bad frame does not result in catastrophic failures. It produces smooth transitions from straight to curves and reduces wobblyness to some extent. 
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There is a lot of scope for improvement in the current pipeline. This pipeline cannot pass both the challenge video and the harder challenge video.

I have assumed that the first frame of the video will give me the position of left and right lane lines and i can do a  highly targeted search for them in the next frame. If the lane lines can't be dectected in the first frame the pipeline simply fails. This can be corrected by analysing the output of sliding window search to decide if the next frame should be searched completely or should we use a targeted search.

The thresholds i have used work only for yellow and white line and then depends on masking out everything outside region of interest to reduce noise. This is not rodust as the region of interest mask has to be tuned if the video changes. A more dynamic thresholding technique needs to be used if we want to make the pipeline more robust.
