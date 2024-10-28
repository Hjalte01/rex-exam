from cv2 import aruco
from examrobot import ExamRobot
from states.calibrate import Calibrate
from states.detect import Detect


ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
MARKER_SIZE = 124.02
BOARD_MARKER_SIZE = 23.32
BOARD_SHAPE = (5, 5)
BOARD_GAP = 1.85

def main():
    robot = ExamRobot()
    calibrate_state = Calibrate(3, ARUCO_DICT, BOARD_MARKER_SIZE, BOARD_SHAPE, BOARD_GAP)
    detect_state = Detect(ARUCO_DICT)
