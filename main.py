from datetime import datetime
from os import path
from time import sleep
import numpy as np
import cv2
from cv2 import aruco
from examrobot import ExamRobot, mock, CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE
from states.calibrate import Calibrate, CalibrateEvent
from states.detect import Detect, DetectEvent
from states.drive import Drive, DriveEvent
from tasks.estimate import Estimate

PI_ENV              = True

# Driver settings
CYCLE               = CYCLE     # 50ms
# Camera settings
IMG_SIZE            = IMG_SIZE  # (1920, 1080)
FPS                 = FPS       # 30
# Grid settings
LANDMARK_SIZE       = LANDMARK_SIZE # 200mm - The size of a landmark (box with marker).
ZONE_SIZE           = ZONE_SIZE     # 450mm
ZONES               = ZONES         # 9
# Aruco settings
MARKER_SIZE         = 145     # mm - The size of a marker on a landmark. Rally marker == 145
BOARD_MARKER_SIZE   = 25.32 # 23.32     # mm - The size of a marker on a board.
BOARD_SHAPE         = (5, 5)    # m x n
BOARD_GAP           = 1.85 # 1.85 #26.77      # mm
ARUCO_DICT          = aruco.Dictionary_get(aruco.DICT_6X6_250)
# Calibrate settings
PASSES              = 30
CONFIG_PATH         = path.abspath("./configs/calibration-2.npz")

def handle_calibrate_complete(e: CalibrateEvent):
    np.savez(
        path.abspath(CONFIG_PATH),
        cam_matrix=e.cam_matrix,
        dist_coeffs=e.dist_coeffs
    )
    e.robot.done(True)

def handle_detect_complete(e: DetectEvent):
    e.robot.switch(Drive.ID)

def handle_drive_complete(e: DriveEvent):
    e.robot.done(True)

def main():
    robot = None
    if PI_ENV:
        robot = ExamRobot(CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE)
    else:
        robot = mock()

    prompt = """
    Press "c" to calibrate.
    Press "p" to capture a picture.
    Press "e" to make an estimation.
    Press "d" to drive to start drive sequence.
    Press "s" to stop.
    Press "q" to quit.\n"""

    while True:
        c = (input(prompt) + "\n").lower()[0]
        
        if c == 'c':
            robot.add(Calibrate(PASSES, ARUCO_DICT, BOARD_MARKER_SIZE, BOARD_SHAPE, BOARD_GAP), default=True)
            robot.register(CalibrateEvent.COMPLETE, handle_calibrate_complete)
            robot.start()
            
            while not robot.done():
                robot.wait_for(CalibrateEvent.PASS_COMPLETE)
            robot.stop()
        elif c == 'p':
            frame = robot.capture()
            cv2.imwrite(
                path.abspath(
                    "./imgs/capture-{0}.png".format(datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
                ),
                frame
            )
        elif c == 'e':
            from tasks.estimate import rvec_to_rmatrix, tvec_to_euclidean
            frame = robot.capture()
            corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), ARUCO_DICT)

            if ids is None:
                continue

            config = np.load(CONFIG_PATH)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 
                MARKER_SIZE*0.001, 
                config["cam_matrix"],
                config["dist_coeffs"]
            )

            print("Delta: {0:>11.3f}mm".format(tvec_to_euclidean(tvecs[0][0])))
            (pitch, yaw, roll) = rvec_to_rmatrix(rvecs[0][0])
            print(
                "Pitch: {0:>11.3f}deg\nYaw: {1:>11.3f}deg\nRoll: {2:>11.3f}".format(
                    np.rad2deg(pitch),
                    np.rad2deg(yaw),
                    np.rad2deg(roll)
                )
            )
        elif c == 'd':
            config = np.load(path.abspath(CONFIG_PATH))
            robot.add(Estimate(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"], robot.grid))
            robot.add(Detect(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"]), default=True)
            robot.add(Drive([1, 4, 3, 2, 1]))
            robot.register(DetectEvent.COMPLETE, handle_detect_complete)
            robot.register(DriveEvent.COMPLETE, handle_drive_complete)
            robot.start()

            while not robot.done():
                robot.wait_for(DriveEvent.GOAL_VISITED)
        elif c == 't':
            robot.grid.create_marker(robot.grid[3, 5].diffuse(), robot.grid[3, 5][3, 3], 8, LANDMARK_SIZE)
            robot.grid.create_marker(robot.grid[3, 3].diffuse(), robot.grid[3, 3][3, 3], 4, LANDMARK_SIZE)
            # robot.grid.create_marker(robot.grid[5, 7].diffuse(), robot.grid[5, 5][3, 3], 7, LANDMARK_SIZE)

            # frame = robot.capture()
            # cv2.imwrite(
            #     path.abspath(
            #         "./imgs/capture-{0}.png".format(datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
            #     ),
            #     frame
            # )

            config = np.load(CONFIG_PATH)
            estimate = Estimate(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"], robot.grid)
            estimate.run(robot)
            # robot.add(Detect(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"]), default=True)
            # robot.register(DetectEvent.COMPLETE, handle_detect_complete)

            # while not robot.done():
            #     robot.wait_for(DetectEvent.COMPLETE)
            # robot.stop()

        elif c == 's':
            robot.stop()
        elif c == 'q':
            robot.stop()
            break
        
if __name__ == "__main__":
    main()
