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
        self.cycle = 0.0
        self.first_theta = None
        self.last_theta = None
        self.first_id = None
    
    def run(self, robot: ExamRobot):
        robot.stop()
        sleep(0.2)
        frame = robot.cam.capture()
        
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.aruco_dict)
        if ids is None:
            robot.go_diff(40, 40, 1, 0)
            sleep(0.1)
            return
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size, 
            self.cam_matrix,
            self.dist_coeffs
        )

        visited = set()
        for i, (rvec, tvec, id) in enumerate(zip(rvecs, tvecs, ids)):
            self.fire(DetectEvent(DetectEvent.DETECTED, id=id))

            orientation = rvec_to_rmatrix(rvec)
            theta = (robot.heading + orientation[1])%(2*np.pi)
            delta = tvec_to_euclidean(tvec)

            if id[0] in visited:
                continue
            visited.add(id[0])

            if self.first_id is None:
                self.first_id = id[0]
                self.first_theta = theta
            elif self.first_id != id[0]:
                self.cycle = (theta - self.first_theta)/self.count
                print(self.cycle)
            
            # all ids unique then go on else "contine" to the next iteration - only include the same marker id once 
            if all(m.id != id[0] for m in robot.grid.markers):
                robot.grid.update(robot.grid.origo, Position(delta, theta), id[0])
                print(len(robot.grid.markers)) 
                print("[LOG] {0} - Detected marker {1}.".format(self, id[0]))

        # if self.first_theta and self.last:
        #     robot.heading = ((self.first_theta - self.last)/2)%(2*np.pi)
        if self.first_theta:
            robot.heading += self.first_theta
            
        self.count += 1
        print(f"count: {np.rad2deg(self.count)}, heading: {np.rad2deg(robot.heading)}")
        if self.first_theta:
            print(f'first: {np.rad2deg(self.first_theta)}')
        else:
            print("First is None?")

        print(self.cycle*self.count)
        if self.count*self.cycle >= 2*np.pi:
            print("[LOG] {0} - Detect complete.".format(self))
            robot.stop()
            self.done(True)
            self.fire(DetectEvent(DetectEvent.COMPLETE))
            return
        
        robot.go_diff(40, 40, 1, 0)
        sleep(0.1)

        self.first_theta = None
            