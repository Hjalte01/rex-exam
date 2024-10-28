import cv2
import numpy as np
from cv2 import aruco
from time import sleep
from examrobot import ExamRobot
from statedriver import Event, EventType, State
from pathplaning.grid import Position

class DetectEvent(Event):
    MARKER_DETECTED = EventType("EVENT-MARKER-DETECTED")
    DETECTION_COMPLETE = EventType("EVENT-DETECTION-COMPLETE")

    def __init__(self, type, **kwords):
        super().__init__(type, **kwords)

def tvec_to_euclidean(v):
    return np.linalg.norm(v)*1000 

def rvec_to_rmatrix(v):
    rmatrix, _ = cv2.Rodrigues(v)
    sy = np.sqrt(rmatrix[0][0]**2 + rmatrix[1][0]**2)

    if sy >= 1e-6: # Non-singular, whatever that is.
        x = np.arctan2(rmatrix[2][1], rmatrix[2][2])
        y = np.arctan2(-rmatrix[2][0], sy)
        z = np.arctan2(rmatrix[1][0], rmatrix[0][0])
    else:
        x = np.arctan2(-rmatrix[1][2], rmatrix[1][1])
        y = np.arctan2(-rmatrix[2][0], sy)
        z = 0
    return [x, y, z]

class Detect(State):
    def __init__(self, aruco_dict):
        super().__init__("STATE_DETECT")
        self.aruco_dict = aruco_dict
        self.first = None
    
    def run(self, robot: ExamRobot):
        robot.stop()
        sleep(0.1)
        frame = robot.camera.capture()
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict)
        
        if ids is None:
            robot.go_diff(40, 40, 1, 0)
            return
        
        rvecs, tvecs = aruco.estimatePoseSingleMarkers(
            corners, 
            robot.marker_size, 
            robot.cam_matrix,
            robot.dist_coeffs
        )

        first, last
        for i, (rvec, tvec, id) in enumerate(zip(rvecs, tvecs, ids)):
            if not all(m.id != id for m in robot.grid.markers):
                continue

            self.fire(DetectEvent(DetectEvent.MARKER_DETECTED, id=id))

            orientation = rvec_to_rmatrix(rvec)
            theta = (robot.heading + orientation[1])%2*np.pi
            delta = tvec_to_euclidean(tvec)
            robot.grid.update(robot.grid.origo, Position(delta, theta), id)

            if i + 1 == len(ids):
                last = theta
            elif not i:
                first = theta

            if self.first is None:
                self.first = theta
            elif theta > self.first:
                self.fire(DetectEvent(DetectEvent.DETECTION_COMPLETE, id=id))
                self.done(True)

        if self.done():
            return
            
        robot.heading = ((first - last)/2)%2*np.pi
        robot.go_diff(40, 40, 1, 0)
            
