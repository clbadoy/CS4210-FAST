# Problem 1: Figuring out how to extract a pixel's Grayscale Value from an image
# Problem 2: Figuring out how to catch pixels @ circle (Rough)
# Problem 3: Figuring out conditions for the circle to be considered a keypoint
# Problem 4: How to Draw the results onto the image with matplotlib
# Problem 5: Too Many Drawings of corners, narrow down if possible with an-NMS

# NMS Steps (Not implemented)
# ----------------------------------
# * For all key point pixel points, compute a score function V.
# * V = Sum of absolute difference between the pixels in the contiguous arc and the center pixel
# * Consider two adjacent points
# * For two adjacent key point pixels (eg key pixel “B” within key pixel “A’s” circle), compare score values
# * Discard the key pixel with the lower V score.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import warnings

# Test to see if pixel is brighter than center pixel
# Returns 1 if true, 0 if false
def brighter_test(pixel, threshold):
    return pixel >= threshold

# Test to see if pixel is darker than center pixel
# Returns 1 if true, 0 if false
def darker_test(pixel, threshold):
    return pixel <= threshold

# Assuming N = 9; existing number of bright/darker pixels
# (Problem 3) Returns (1) if 9 or more arc pixels are either brighter/darker than key pixel
def corner_test(image, center, pixels, threshold, contiguous_ct):
    
    # Filter out excess console spam of overflow.
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in scalar subtract")

    bright_count = 0
    dark_count = 0
    max_bright_count = 0
    max_dark_count = 0


    v_bright = 0
    v_dark = 0
    v_bright_max = 0
    v_dark_max = 0

    for pixel_coords in pixels:
        pixel_value = image[pixel_coords[0], pixel_coords[1]]

        if brighter_test(pixel_value, center + threshold):
            bright_count += 1
            dark_count = 0
            max_bright_count = max(max_bright_count, bright_count)
            
            v_bright += abs(pixel_value - center)
            v_dark = 0
            v_bright_max = max(v_bright_max, v_bright)

        elif darker_test(pixel_value, center - threshold):
            dark_count += 1
            bright_count = 0
            max_dark_count = max(max_dark_count, dark_count)

            v_dark += abs(pixel_value - center)
            v_bright = 0
            v_dark_max = max(v_dark_max, v_dark)
        else:
            bright_count = 0
            dark_count = 0

            v_bright = 0
            v_dark = 0
    return max_bright_count >= contiguous_ct or max_dark_count >= contiguous_ct, max(v_bright_max, v_dark_max)

# (Problem 2) Getting values of each pixel in a circle w/in a 3 pixel radius...
# TODO [Done] Unsure how to NOT hardcode a circle and make it dynamically accept different radii
# TODO [Done] Necessary to handle outside image borders? (Doesn't test all pixels, but pixels within a border)

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# TODO Potential room for speedup?
def euclidean_circle_border(image, center, radius):
    border_pixels = []
    row, col = image.shape

    # Bounding Box
    x_min = max(0, int(center[0] - radius))
    x_max = min(row - 1 , int(center[0] + radius))
    y_min = max(0, int(center[1] - radius))
    y_max = min(col - 1, int(center[1] + radius))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            distance = euclidean_distance((x, y), center)
            if abs(distance - radius) < 1:
                border_pixels.append((x, y))
    #print(len(border_pixels))
    return border_pixels
                
'''# ----------- Serial Version --------------'''
def circle(image, radius, threshold, contiguous):
    # Gets Length & Width of Image
    row, col = image.shape
    keypoints_coord = []
    v_score_arr = []
    

    for x in range(0, row):
        for y in range(0, col):
            px_center = image[x, y]
            px_arc = []
            
            # im[x,y] yields out a pixel's grayscale value
            # If RGB, then im[x,y] yields RGB values
            px_arc = euclidean_circle_border(image, (x, y), radius)

            #print("(", x, ",", y, ")", px_center, px_arc) # Validate pixel values spitting up different numbers

            corner_bool, v_score = corner_test(image, px_center, px_arc, threshold, contiguous)
            # Tests for main keypoints
            if corner_bool:
                keypoints_coord.append((x, y))
                v_score_arr.append(v_score)
                #print(calc_v_score(px_center, px_arc, threshold)) # V Score Maker for NMS
    
    return keypoints_coord, v_score_arr

'''# ---------MultiProcessing version----------'''
def circle_process(image, start_row, end_row, radius, threshold, contiguous, res_glob_manager):
    keypoints_coord = []
    v_score_arr = []
    
    for x in range(start_row, end_row):
        for y in range(image.shape[1]):
            px_center = image[x, y]
            px_arc = []
            
            # im[x,y] yields out a pixel's grayscale value
            # If RGB, then im[x,y] yields RGB values
            px_arc = euclidean_circle_border(image, (x, y), radius)

            #print("(", x, ",", y, ")", px_center, px_arc) # Validate pixel values spitting up different numbers

            corner_bool, v_score = corner_test(image, px_center, px_arc, threshold, contiguous)
            # Tests for main keypoints
            if corner_bool:
                keypoints_coord.append((x, y))
                v_score_arr.append(v_score)

    res_glob_manager.append((keypoints_coord, v_score_arr))


# (Problem 4) Draws the image with keypoomts 
def display_image(image, keypoints):
    image_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    for keypoint in keypoints:
        cv2.circle(image_display, (keypoint[1], keypoint[0]), 1, (0, 255, 0) , -1) # (image, center_coords, radius, color, thickness)

    cv2.imwrite("FastTest-NMS.jpg", image_display)
    plt.imshow(image_display)
    plt.show()

# (Problem 5) NMS
def NMS(keypoints, v_scores, radius):
    filtered_keypoints = []
    num_keypoints =  len(keypoints)

    discarded = np.zeros(num_keypoints, dtype=bool)

    for i in range(num_keypoints):
        if not discarded[i]:
            curr_keypoint = keypoints[i]
            curr_v_score = v_scores[i]
            filtered_keypoints.append(curr_keypoint)

            for j in range(i + 1, num_keypoints):
                if not discarded[j]:
                    next_keypoint = keypoints[j]
                    next_v_score = v_scores[j]

                    distance = euclidean_distance(curr_keypoint, next_keypoint)

                    if distance < radius:
                        if next_v_score < curr_v_score:
                            discarded[j] = True
    return filtered_keypoints

# OpenCV's implementation of FAST ; Returns (x, y) tuple of keypoint locations.
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
#def OpenCV_FAST(image):

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img, None)

    # Accessing keypoint coordinates
    keypoint_coordinates = [kp[i].pt for i in range(len(kp))]
    print("Keypoint Coordinates:", keypoint_coordinates)

    img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

    cv2.imwrite('fast_true.png', img2)

    '''# Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv2.imwrite('fast_false.png', img3)
    plt.imshow(img2)
    plt.show()'''

    return keypoint_coordinates

def mean_square_error(keypoint_custom, keypoint_openCV):
    total_mse = 0
    num_matches = min(len(keypoint_custom), len(keypoint_openCV))

    for i in range(num_matches):
        kp1 = keypoint_custom[i]
        kp2 = keypoint_openCV[i]

        mse_x = (kp1[0] - kp2[0])**2
        mse_y = (kp1[1] - kp2[1])**2

        mse = (mse_x + mse_y) / 2

        total_mse += mse

    return total_mse / num_matches

if __name__ == '__main__':

    print("-" * 100)
    print("Program Start: Multiprocessing.")
    print("-" * 100)
    print("\n")
    # ----------- (Problem 1) Load in the Image & Convert to Grayscale -----------
    # Testing a 480x480 image
    start_time = time.time()
    manager = multiprocessing.Manager()
    image = cv2.imread('Pink_flower.jpg', cv2.IMREAD_GRAYSCALE)

    # ----------- Configurations for FAST -----------
    # contiguous_requirement: Number of contiguous pixels to be considered a keypoint
    # threshold_value: Brightness difference threshold required to be brighter or darker
    # radius: circle size of FAST
    # process_count: How many processes to use for FAST calculations in multiprocessing
    # ----------------------------------------------
    contiguous_requirement = 9
    threshold_value = 16
    radius = 3
    process_count = 12 # Max is 12 for Intel Core-i7 10750H

    # ---------- Multiprocessing initialization for FAST ----------

    # Divide into equal rows
    row_steps = image.shape[0] // process_count
    processes = []
    result_global_manager = manager.list()

    keypoints_all = []
    v_scores_all = []

    for i in range(process_count):
        
        start_row = i * row_steps
        if i < process_count - 1:
            end_row = start_row + row_steps
        else:
            end_row = image.shape[0]
        
        # Each process responsible to conduct circle_process function per equally divided rows.
        process = multiprocessing.Process(target = circle_process, args = (image, start_row, end_row, radius, threshold_value, contiguous_requirement, result_global_manager))
        processes.append(process)
        process.start()

    print("-" * 100)
    print("Checkpoint 1: Processes are all running for FAST calculations.")
    print("-" * 100)
    print("\n")
    for process in processes:
        process.join()

    print("-" * 100)
    print("Checkpoint 2: Processes finished Running for FAST calculations. Merging data from processes...")
    print("-" * 100)
    # Collect Potential Keypoints of the Image with attached v_score
    for i in result_global_manager:
        keypoints_all.extend(i[0])
        v_scores_all.extend(i[1])
    
    # Clearing list for reuse
    processes.clear()
    result_global_manager[:] = []

    # Serial Function (Kept for documentation reasons)
    # keypoints, v_combined = circle(image, radius, threshold_value, contiguous_requirement)

    #keypoints_NMS = NMS(keypoints, v_combined, radius)

    time_elapsed = time.time() - start_time
    print("\n")
    print("-" * 100)
    print("-" * 100)
    print("Contiguous Requirement: %d" %(contiguous_requirement))
    print("Threshold: %d " % (threshold_value))
    print("Radius: %d" % (radius))

    print("Time Elapsed: %.5f seconds" % (time_elapsed))
    #print("Total Keypoints with nonMaxSuppression: %d\nTotal Keypoints without nonMaxSuppression: %d" %(len(keypoints_all), len(keypoints_all)))
    print("Total Keypoints without nonMaxSuppression: %d" %(len(keypoints_all)))
    print("-" * 100)
    print("-" * 100)

    # Show Image with Matlib (No NMS)

    display_image(image, keypoints_all)