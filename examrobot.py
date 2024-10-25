import numpy as np
from robot import Robot
from camera import Camera
from pathplaning.grid import Grid, Position
from pathplaning.localization import ParticleFilter

class ExamRobot(Robot):
    def __init__(self, port='/dev/ttyACM0'):
        super().__init__(port)
        self.heading = 0.5*np.pi
        self.grid = Grid((0, 0), 450, 9, 200)
        self.pf = ParticleFilter(Position(np.sqrt(2*225**2), 0.25*np.pi), 1000, self.grid) 
        self.camera = Camera((1920, 1080), 30, Camera.Strategy.PI_CAMERA_REQ)
        self.aruco_dict = None
        self.marker_length = 0.0
        self.cam_matrix = None
        self.dist_coeffs = None

    def __del__(self):
        return super().__del__()
    