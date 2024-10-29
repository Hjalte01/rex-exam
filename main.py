from datetime import datetime
from os import path
import numpy as np
import cv2
from cv2 import aruco
from examrobot import ExamRobot, mock, CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE
from states.calibrate import Calibrate, CalibrateEvent
from states.detect import Detect
from tasks.estimate import Estimate

PI_ENV              = False

# Driver settings
CYCLE               = CYCLE # 50ms
# Camera settings
IMG_SIZE            = IMG_SIZE  # (1920, 1080)
FPS                 = FPS       # 30
# Grid settings
LANDMARK_SIZE       = LANDMARK_SIZE # 200mm - The size of a landmark (box with marker).
ZONE_SIZE           = ZONE_SIZE     # 450mm
ZONES               = ZONES         # 9
# Aruco settings
MARKER_SIZE         = 124.02    # mm - The size of a marker on a landmark.
BOARD_MARKER_SIZE   = 41.80     # mm - The size of a marker on a board.
BOARD_SHAPE         = (5, 5)    # m x n
BOARD_GAP           = 2.48      # mm
ARUCO_DICT          = aruco.Dictionary_get(aruco.DICT_6X6_250)
# Calibrate settings
PASSES              = 3

def handle_calibrate_event(event: CalibrateEvent):
    np.savez(
        path.abspath("./configs/calibration.npz"),
        cam_matrix=event.cam_matrix,
        dist_coeffs=event.dist_coeffs
    )

def main():
    # robot = ExamRobot(CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE)
    robot = None
    if PI_ENV:
        robot = ExamRobot(CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE)
    else:
        robot = mock()
    robot.add(Estimate(ARUCO_DICT, (0, 0)))
    detect_state = Detect(ARUCO_DICT)
    calibrate_state = Calibrate(PASSES, ARUCO_DICT, BOARD_MARKER_SIZE, BOARD_SHAPE, BOARD_GAP)
    robot.add(detect_state, default=True)
    robot.add(calibrate_state)
    robot.register(CalibrateEvent.CALIBRATE_COMPLETE, handle_calibrate_event)

    prompt = """
    Press "c" to calibrate.
    Press "p" to capture a picture.
    Press "e" to make an estimation.
    Press "d" to drive to start drive sequence.
    Press "s" to stop.
    Press ESC to quit."""

    while True:
        c = (input(prompt) + "").lower()[0]

        if c == 0x63:
            robot.switch(Calibrate.ID)
            robot.start()

            for i in range(PASSES):
                print(f"Calibration pass {i + 1} of {PASSES}.")
                robot.wait_for(CalibrateEvent.PASS_COMPLETE)

                c = (input("\tPress \"c\" to continue.\n\tPress ESC to stop.\n") + "").lower()[0]
                if c == 0x1B:
                    break
        elif c == 0x70:
            frame = robot.capture()
            cv2.imwrite(
                path.abspath(
                    "./imgs/capture-{0}.png".format(datetime.now().strftime('%Y%m%dT%H%M%S'))
                ),
                frame
            )
        elif c == 0x65:
            from tasks.estimate import rvec_to_rmatrix, tvec_to_euclidean
            frame = robot.capture()
            corners, ids, _ = aruco.detectMarkers(frame, ARUCO_DICT)

            if ids is None:
                continue

            rvecs, tvecs = aruco.estimatePoseSingleMarkers(
                corners, 
                robot.marker_size, 
                robot.cam_matrix,
                robot.dist_coeffs
            )

            print("Delta: {0>11.3f}mm", tvec_to_euclidean(tvecs[0]))
            (pitch, yaw, roll) = rvec_to_rmatrix(rvecs[0])
            print(
                "Pitch: {0>11.3f}deg\nYaw: {0>11.3f}deg\nRoll: {0>11.3f}".format(
                    np.rad2deg(pitch),
                    np.rad2deg(yaw),
                    np.rad2deg(roll)
                )
            )
        elif c == 0x64:
            robot.start()
        elif c == 0x73:
            robot.stop()
        elif c == 0x1B:
            robot.stop()
            break
        
if __name__ == "__main__":
    main()
