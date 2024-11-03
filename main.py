from datetime import datetime
from os import path
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
MARKER_SIZE         = 92.12     # mm - The size of a marker on a landmark. Rally marker == 145
BOARD_MARKER_SIZE   = 42.37     # mm - The size of a marker on a board.
BOARD_SHAPE         = (4, 4)    # m x n
BOARD_GAP           = 5.34 #26.77      # mm
ARUCO_DICT          = aruco.Dictionary_get(aruco.DICT_6X6_250)
# Calibrate settings
PASSES              = 14

def handle_calibrate_pass_complete(e: CalibrateEvent):
    e.origin.wait()

def handle_calibrate_complete(e: CalibrateEvent):
    np.savez(
        path.abspath("./configs/calibration.npz"),
        cam_matrix=e.cam_matrix,
        dist_coeffs=e.dist_coeffs
    )

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
            started = False
            robot.add(Calibrate(PASSES, ARUCO_DICT, BOARD_MARKER_SIZE, BOARD_SHAPE, BOARD_GAP), default=True)
            robot.register(CalibrateEvent.COMPLETE, handle_calibrate_complete)
            robot.register(CalibrateEvent.PASS_COMPLETE, handle_calibrate_pass_complete)

            for i in range(PASSES):
                c = (input(
                        "Calibration pass {0} of {1}.\n" 
                        "Press \"c\" to continue.\n"
                        "Press \"q\" to stop.".format(i + 1, PASSES)) + "\n"
                    ).lower()[0]
                if c == 'q':
                    break

                if not started:
                    robot.start()
                    started = True
                else:
                    robot.wake()

                robot.wait_for(CalibrateEvent.PASS_COMPLETE)
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

            config = np.load(path.abspath("./configs/calibration.npz"))
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
            config = np.load(path.abspath("./configs/calibration.npz"))
            robot.add(Estimate(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"], [np.sqrt(100**2), 0], robot.grid))
            robot.add(Detect(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"]), default=True)
            robot.add(Drive([1, 4, 3, 2, 1]))
            robot.register(DetectEvent.COMPLETE, handle_detect_complete)
            robot.register(DriveEvent.COMPLETE, handle_drive_complete)
            robot.start()

            while not robot.done():
                robot.wait_for(DriveEvent.GOAL_VISITED)
        elif c == 't':
            config = np.load(path.abspath("./configs/calibration.npz"))
            robot.add(Estimate(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"], (np.sqrt(90**2), 0), robot.grid))
            robot.add(Detect(ARUCO_DICT, MARKER_SIZE, config["cam_matrix"], config["dist_coeffs"]), default=True)
            # robot.add(Drive([11]))
            robot.register(DetectEvent.COMPLETE, lambda e: e.robot.done(True))
            robot.register(DetectEvent.COMPLETE, handle_detect_complete)
            robot.register(DriveEvent.COMPLETE, handle_drive_complete)
            robot.start()

            while not robot.done():
                robot.wait_for(DetectEvent.COMPLETE)
            robot.driver.stop()
        elif c == 's':
            robot.stop()
        elif c == 'q':
            robot.stop()
            break
        
if __name__ == "__main__":
    main()
