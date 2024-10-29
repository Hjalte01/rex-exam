from typing import Tuple
import numpy as np
from cv2 import aruco
from statedriver import EventType, Event, State
from examrobot import ExamRobot

class CalibrateEvent(Event):
    PASS_COMPLETE = EventType("EVENT-CALIBRATE-PASS-COMPLETE")
    CALIBRATE_COMPLETE = EventType("EVENT-CALIBRATE-COMPLETE")

    def __init__(self, type, **kwords):
        super().__init__(type, **kwords)

class Calibrate(State):
    ID = "STATE_CALIBRATE"

    def __init__(self, passes: int, aruco_dict, marker_length: float, board_shape: Tuple[int, int], board_gap: float):
        super().__init__(Calibrate.ID)
        self.passes = passes
        self.corners =  np.empty((0, 1, 4, 2), np.float32)
        self.ids = np.empty((0, 1), np.int32)
        self.counts = np.empty((0, 1), np.int32)
        self.aruco_dict = aruco_dict
        self.board = aruco.GridBoard.create(
            board_shape[0], board_shape[1], marker_length*0.001, board_gap*0.001, aruco_dict
        )

    def run(self, robot: ExamRobot):
        frame = robot.cam.capture()        
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict)

        if ids is None:
            return
        
        self.corners = np.append(self.corners, corners, axis=0)
        self.ids = np.append(self.ids, ids, axis=0)
        self.counts = np.append(self.counts, [len(ids)])
        
        self.passes -= 1
        if self.passes > 0:
            self.fire(CalibrateEvent(CalibrateEvent.PASS_COMPLETE))
            return
        
        _, cam_matrix, dist_coeffs, _, _ = aruco.calibrateCameraAruco(
            self.corners,
            self.ids,
            self.counts,
            self.board,
            robot.cam.img_size,
            None,
            None
        )

        self.fire(
            CalibrateEvent(CalibrateEvent.CALIBRATE_COMPLETE),
            cam_matrix=cam_matrix,
            dist_coeffs=dist_coeffs
        )
