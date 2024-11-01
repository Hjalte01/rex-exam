from datetime import datetime
from os import path
import cv2
import numpy as np
from cv2 import aruco
from time import sleep
from examrobot import ExamRobot
from statedriver import Event, EventType, State
from pathplaning.grid import Position

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
        self.count = 0
        self.first = None
        self.last = None
    
    def run(self, robot: ExamRobot):
        robot.stop()
        sleep(0.2)
        print("a")
        frame = robot.cam.capture()
        print("b")
        # cv2.imwrite(
        #     path.abspath(
        #         "./imgs/detect-{0}.png".format(datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
        #     ),
        #     frame
        # )
        
        corners, ids, _ = aruco.detectMarkers(frame, self.aruco_dict)
        if ids is None:
            robot.go_diff(40, 40, 1, 0)
            sleep(0.2)
            return
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size, 
            self.cam_matrix,
            self.dist_coeffs
        )

        for i, (rvec, tvec, id) in enumerate(zip(rvecs, tvecs, ids)):
            # robot.log_file.write("[LOG] {0} - Detected marker {1}.".format(self, id))
            self.fire(DetectEvent(DetectEvent.DETECTED, id=id))

            orientation = rvec_to_rmatrix(rvec)
            theta = (robot.heading + orientation[1])%(2*np.pi)
            delta = tvec_to_euclidean(tvec)

            if i + 1 == len(ids):
                self.last = theta
            elif not i:
                self.first = theta

            # all ids unique then go on else "contine" to the next iteration - only include the same marker id once 
            if all(m.id != id for m in robot.grid.markers):
                robot.grid.update(robot.grid.origo, Position(delta, theta), id)
                print(robot.grid.markers)   
                print("[LOG] {0} - Detected marker {1}.".format(self, id))


        if self.first and self.last:
            robot.heading = ((self.first - self.last)/2)%(2*np.pi)
        elif self.first:
            robot.heading += self.first
        print(f"count: {self.count}, first: {self.first}, last: {self.last}, heading: {robot.heading}")
        
        self.count += robot.heading
        if self.count >= 2*np.pi:
            print("[LOG] {0} - Detect complete.".format(self))
            robot.stop()
            self.done(True)
            self.fire(DetectEvent(DetectEvent.COMPLETE))
            return
        
        robot.go_diff(40, 40, 1, 0)
        sleep(0.2)

        self.first = None
        self.last = None
            