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
        self.cycle_theta = 100 # a cycle isn't 30 
        self.first_theta = 0.0
        self.first_id = None
        self.map = dict()
    
    def run(self, robot: ExamRobot):
        robot.stop()
        sleep(0.2)
        
        if self.cycle_theta != 100 and self.count*self.cycle_theta >= 2*np.pi:
            print("[LOG] {0} - Detect complete.".format(self))
            self.done(True)
            self.fire(DetectEvent(DetectEvent.COMPLETE))
            print(", ".join([m.__str__() for m in robot.grid.markers]))
            return

        frame = robot.cam.capture()
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.aruco_dict)
        if ids is None:
            self.count += 1
            robot.heading = self.count*self.cycle_theta
            robot.go_diff(40, 40, 1, 0)
            print(f"heading: {np.rad2deg(robot.heading)}")
            print(f"count: {self.count}, cycle_theta: {self.cycle_theta}")
            sleep(0.01)
            return
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size*0.001, 
            self.cam_matrix,
            self.dist_coeffs
        )

        for (rvec, tvec, id) in zip(rvecs, tvecs, ids):
            
            orientation = rvec_to_rmatrix(rvec)

            if any(m.id == id[0] for m in robot.grid.markers):
                print(f"self.map[id[{id[0]}]] = {orientation[1]}")
                self.map[id[0]].append(orientation[1])
                continue
            self.fire(DetectEvent(DetectEvent.DETECTED, id=id[0]))

            theta = robot.heading + orientation[1]
            delta = tvec_to_euclidean(tvec)
            print("delta, ", delta)
            print(f"self.map[id[{id[0]}]] = {orientation[1]}")
            self.map.setdefault(id[0], [orientation[1]])


            robot.grid.update(robot.grid.origo, Position(delta, theta % (2 * np.pi)), id[0])
            print("[LOG] {0} - Detected marker {1}.".format(self, id[0]))

        sum_delta = 0
        n_delta = 0
        for _, orientations in self.map.items():
            if len(orientations) < 2:
                continue
            for index in range(len(orientations)-1):
                sum_delta += orientations[index] - orientations[index+1]
            n_delta += len(orientation)-1
            # if delta < self.cycle_theta:
        delta = sum_delta / n_delta
        self.cycle_theta = delta


        self.count += 1
        robot.heading = self.count*self.cycle_theta
        robot.go_diff(40, 40, 1, 0)
        # print(f"heading: {np.rad2deg(robot.heading)}")
        # print(f"count: {self.count}, cycle_theta: {self.cycle_theta}")
        sleep(0.01)

            