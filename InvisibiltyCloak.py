# 1. Capture and store the background frame. [ This will be done for some seconds ]
# 2. Detect the red colored cloth using color detection and segmentation algorithm
# 3. Segment out the red colored cloth by generating a mask. [ used in code ]
# 4. Generate the final augmented output to create a magical effect. [ video.mp4 ]

import cv2
import time
import numpy as np

# To save the output in a file output.avi.
# Fourcc is a four byte code used to specify the video codec.
# For more check is https://www.fourcc.org/

fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# This is for starting the webcam

cap = cv2.VideoCapture(0)

# It is for make the code sleep for 2 seconds and allow the webcam to start

time.sleep(2)

bg = 0

# Capturing the background for 60 frames

for i in range(0, 60):
    ret, bg = cap.read()
    bg = np.flip(bg, axis = 1)

# it is for reading the captured frame until the camera is open.

while(cap.isOpened()):

    # Using the cap.isOpened() to check if the camera is open or not
    # ret returns a boolean value if it is true or false

    ret, img = cap.read()
    if(not ret):
        break

    #Flipping the image for consistency

    img = np.flip(img, axis = 1)

    # converting the color to bgr(blue, green, red) to hsv(hue, saturation, value)
    # to detect the red color more efficiently
    # Hue: This channel encodes color information. Hue can be thought of as an angle where 0 degree corresponds to the red color, 120 degrees 
    # corresponds to the green color, and 240 degrees corresponds to the blue color
    # Saturation: This channel encodes the intensity/purity of color. For example, pink is less saturated than red.
    # Value: This channel encodes the brightness of color. Shading and gloss components of an image appear in this channel reading the 
    # videocapture video

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generating ask to detect red color
    # We are creating two different masks which will help us detect the color as per that range

    lower_red = np.array([0, 150, 50])

    upper_red = np.array([10, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])

    upper_red = np.array([180, 255, 255])

    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1+mask_2

    # Open and expand the image where there is mask1(color)
    # Morphology has 4 synataxes - 
    # morphologyEx(src, dst, op, kernel) This method accepts the following parameters:
        # ● src − An object representing the source (input) image.
        # ● dst − object representing the destination (output) image.
        # ● op − An integer representing the type of the Morphological operation.
        # ● kernel − A kernel matrix
    # morphologyEx() is the method of the class Img Processing which is used to perform operations on a given image.
    # MORPH_OPEN and MORPH_DILATE are two types of effect
    # for more about MORPH https://www.tutorialspoint.com/opencv/opencv_morphological_operations.htm

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Selcting only the part that doesn't have mask one and saving in mask two.

    mask_2 = cv2.bitwise_not(mask_1)
    
    # Keeping only the part of the images without the red color

    res_1 = cv2.bitwise_and(img, img, mask = mask_2)

    # Keeping only the part of the images with the red color

    res_2 = cv2.bitwise_and(bg, bg, mask = mask_1)

    # Generating the final output by merging result one and result two

    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)

    output_file.write(final_output)

    # Displaying the output to the user

    cv2.imshow('Magic', final_output)

    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()