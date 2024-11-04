import cv2
import numpy as np
from cv2 import aruco
from time import sleep
from examrobot import ExamRobot
from statedriver import Event, EventType, State
from pathplaning.grid import Position

K_THETA = 0.8

class DetectEvent(Event):
    DETECTED = EventType("EVENT-DETECT-DETECTED")
    COMPLETE = EventType("EVENT-DETECT-COMPLETE")

    def __init__(self, type, **kwords):
        super().__init__(type, **kwords)
        self.robot: ExamRobot = None

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
    ID = "STATE_DETECT"

    def __init__(self, aruco_dict, marker_size, cam_matrix, dist_coeffs):
        super().__init__(Detect.ID)
        self.aruco_dict = aruco_dict
        self.marker_size = marker_size
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.last_heading = 0.0
        self.theta = 0.0
        
    def run(self, robot: ExamRobot):
        robot.stop()
        sleep(0.2)
        
        if self.theta >= 2*np.pi:
            print("[LOG] {0} - Detect complete.".format(self))
            self.done(True)
            self.fire(DetectEvent(DetectEvent.COMPLETE))
            print(", ".join([m.__str__() for m in robot.grid.markers]))
            return

        frame = robot.cam.capture()
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.aruco_dict)
        if ids is None or len(ids) < 2:
            if len(robot.grid.markers) < 2:
                self.theta += K_THETA
                robot.heading = self.theta
                print(self.theta)
            robot.go_diff(40, 40, 0, 1)
            sleep(0.005)
            return
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size*0.001, 
            self.cam_matrix,
            self.dist_coeffs
        )

        for (rvec, tvec, id) in zip(rvecs, tvecs, ids):
            orientation = rvec_to_rmatrix(rvec)
            if any(m.id == int(id[0]) for m in robot.grid.markers):
                continue
            
            self.fire(DetectEvent(DetectEvent.DETECTED, id=id[0]))
            theta = robot.heading + orientation[1]
            delta = tvec_to_euclidean(tvec)
            robot.grid.update(robot.grid.origo, Position(delta, theta % (2 * np.pi)), int(id[0]))
            print("[LOG] {0} - Detected marker {1}.".format(self, id[0]))

        # PF estimate
        self.theta += robot.heading - self.last_heading
        print(self.theta)
        self.last_heading = robot.heading
        
        robot.go_diff(40, 40, 0, 1)
        sleep(0.005)

            