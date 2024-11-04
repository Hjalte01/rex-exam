# This file is used to find the landmarks (Aruco Marker) in the image plane and drive the robot to the landmark 
# by computing the distance and angle between the robot and the landmark.

import cv2 # Import the OpenCV library
from cv2 import aruco
import time
import sys
import numpy as np
from pprint import *


import os, sys
from time import sleep

# get current directory
currentdir = os.path.dirname(os.path.realpath(__file__))
# get parent directory
parentdir = os.path.dirname(currentdir)
# add parent directory to path
sys.path.append(parentdir)

from robot import Robot

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

# print("OpenCV version = " + cv2.__version__)

# Open a camera device for capturing
imageSize = (1920, 1080)
FPS = 30
cam = picamera2.Picamera2()
frame_duration_limit = int(1/FPS * 1000000) # Microseconds
# Change configuration to set resolution, framerate
picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
                                                            controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
                                                            queue=False)
cam.configure(picam2_config) # Not really necessary
cam.start(show_preview=False)

time.sleep(1)  # wait for camera to setup

    
# Capture an image from the camera
image = cam.capture_array("main")
image_width = image.shape[1]
image_height = image.shape[0]

# Get the camera matrix and distortion coefficients
cam_matrix = np.zeros((3, 3))
coeff_vector = np.zeros(5)

focal_length = 1694.0
principal_point = (image_width / 2, image_height / 2)

cam_matrix[0, 0] = focal_length  # f_x
cam_matrix[1, 1] = focal_length  # f_y
cam_matrix[0, 2] = principal_point[0]  # c_x
cam_matrix[1, 2] = principal_point[1]  # c_y
cam_matrix[2, 2] = 1.0

marker_length = 0.145 # meters

# get the dictionary for the aruco markers
img_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) # As per the assignment


def get_landmark(marker_id, cam, img_dict, cam_matrix, coeff_vector, marker_length):
    """Get the landmark from the camera and return the distance and angle between the robot and a specific landmark"""
    sleep(0.5)
    # Capture an image from the camera
    image = cam.capture_array("main")

    # Detect the markers in the images
    corners, ids, _ = aruco.detectMarkers(image, img_dict)

    # check if the wanted marker is in the detected markers
    print("ids: ", ids)
    if ids is not None and marker_id in ids:
        # Get the index of the wanted marker
        index = np.where(ids == marker_id)
        corners = np.array(corners)[index]

        # Estimate the pose of the markers
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cam_matrix, coeff_vector)

        # Calculate the distance and angle between the robot and the landmark
        distance = np.linalg.norm(tvecs[0])
        angle = np.arctan2(tvecs[0][0][0], tvecs[0][0][2])

        return distance, angle
    else:
        return None, None


def search_for_landmark(marker_id, cam, img_dict, cam_matrix, coeff_vector, marker_length, arlo, leftSpeed, rightSpeed):
    """
    Turn around until the landmark is found and return the distance and angle between the robot and the landmark
    """
    while True:
        distance, angle = get_landmark(marker_id, cam, img_dict, cam_matrix, coeff_vector, marker_length)
        if distance != None:
            return distance, angle
        else:
            # Turn around
            arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
            sleep(0.1)
            arlo.stop()
    

# Correct the angle of the robot while driving towards the landmark
def correct_angle(marker_id, angle, arlo, leftSpeed, rightSpeed):
    """
    Correct the angle of the robot by stopping and turning towards the landmark.
    """
    print("angle: ", angle)
    orientation_threshold = 0.2
    if angle != None:
        while abs(angle) >= orientation_threshold:
            arlo.stop()
            # Get angle between the robot and the landmark
            distance, angle = get_landmark(marker_id, cam, img_dict, cam_matrix, coeff_vector, marker_length)
            if angle == None:
                break
            # Turn left correction
            if angle > orientation_threshold:
                print("Turning left")
                arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
                sleep(0.02)
            # Right turn correction
            elif angle < orientation_threshold:
                print("Turning right")
                arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
                sleep(0.02)
 


def drive_towards_landmark(marker_id, distance, angle, arlo, leftSpeed, rightSpeed):
    """
    Drive the robot towards the landmark and keep updating the distance and angle between the robot and the landmark
    """
    # Turn correction
    correct_angle(marker_id, angle, arlo, leftSpeed, rightSpeed)

    # # Drive towards the landmark
    # while distance > 0.50:
    #     print("Driving towards landmark, distance: ", distance)
    #     arlo.go_diff(leftSpeed, rightSpeed, 1, 1)

    #     # Update the distance and angle between the robot and the landmark
    #     distance, angle = get_landmark(marker_id, cam, img_dict, cam_matrix, coeff_vector, marker_length)

    #     if distance == None:
    #         break

    #     # Turn correction
    #     correct_angle(marker_id, angle, arlo, leftSpeed, rightSpeed)

    # Drive towards the landmark
    arlo.move(distance/2)
    correct_angle(marker_id, angle, arlo, leftSpeed, rightSpeed)
    arlo.move(distance/2)

    print("distance: ", distance)

    
    arlo.stop()
    print("Landmark reached")

    return distance, angle


def main():
    # initialize the robot
    arlo = Robot()

    # sleep for 2 seconds
    sleep(2)


    # set the speed of the robot
    left_motor_diff = 0.920
    leftSpeed = 40*left_motor_diff
    rightSpeed = 40

    # Wanted landmarks to visit
    landmark = 8

    # for landmark in wanted_landmarks:
    # Get the distance and angle between the robot and the landmark
    print("Searching for landmark: ", landmark)
    distance, angle = search_for_landmark(landmark, cam, img_dict, cam_matrix, coeff_vector, marker_length, arlo, leftSpeed, rightSpeed)
    print("Distance: ", distance)
    print("Angle: ", angle)

    # Drive towards the landmark
    print("Driving towards landmark: ", landmark)
    drive_towards_landmark(landmark, distance, angle, arlo, leftSpeed, rightSpeed)

    cam.stop()
    arlo.stop()

main()