from typing import Tuple
import numpy as np
import cv2
from cv2 import aruco
from pathplaning.grid import Position, Grid
from pathplaning.localization import ParticleFilter
from statedriver import Task
from examrobot import ExamRobot

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

class Estimate(Task):
    def __init__(self, aruco_dict, marker_size, cam_matrix, dist_coeffs, initial_control: Tuple[float, float], grid: Grid):
        super().__init__()
        self.aruco_dict = aruco_dict
        self.marker_size = marker_size
        self.cam_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs
        self.control = initial_control
        self.particles = 5000
        self.pf = ParticleFilter(self.particles, grid)
        self.first = None
        self.last = None

    def run(self, robot: ExamRobot):
        frame = robot.cam.capture()
        corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.aruco_dict)

        if ids is None or len(ids) < 2: # Python might not shortcircuit
            return
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 
            self.marker_size*0.001, 
            self.cam_matrix,
            self.dist_coeffs
        )

        poses = []
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            orientation = rvec_to_rmatrix(rvec)
            theta = (robot.heading + orientation[1])%2*np.pi
            delta = tvec_to_euclidean(tvec)
            poses.append(Position(delta, theta))

            if i + 1 == len(ids):
                self.last = theta
            elif not i:
                self.first = theta

        x1, y1 = robot.grid.origo.cx, robot.grid.origo.cy
        (x, y), heading = self.pf.update(self.control, poses)
        robot.grid.update(robot.grid.transform_xy(x, y))
        robot.heading = heading

        x2, y2 = robot.grid.origo.cx, robot.grid.origo.cy
        delta = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.control[0] = delta
        self.control[1] = heading
