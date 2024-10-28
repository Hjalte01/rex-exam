from typing import Callable
import numpy as np
from robot import Robot
from camera import Camera
from statedriver import Driver, Event, State, Task
from pathplaning.grid import Grid, Position
from pathplaning.localization import ParticleFilter

class ExamRobot(Robot):
    def __init__(self, cycle=50, img_size=(1920, 1080), fps=30, zone_size=450, zones=9, marker_size=200, port='/dev/ttyACM0'):
        super().__init__(port)
        self.driver = Driver(self, cycle)
        self.heading = 0.5*np.pi
        self.grid = Grid((0, 0), zone_size, zones, marker_size)
        # self.pf = ParticleFilter(Position(np.sqrt(2*(zone_size/2)**2), 0.25*np.pi), 1000, self.grid) 
        self.camera = Camera(img_size, fps, Camera.Strategy.PI_CAMERA_REQ)
        self.marker_size = marker_size
        self.cam_matrix = None
        self.dist_coeffs = None

    def __del__(self):
        return super().__del__()
    
    def capture(self):
        return self.camera.capture()
    
    def default(self, state: State):
        self.driver.default(state)

    def add(self, state: State, default=False):
        return self.driver.add(state, default)

    def switch(self, id):
        self.driver.switch(id)

    def register(self, id, listener: Callable[[Event], None]):
        self.driver.register(id, listener)

    def task(self, task: Task):
        self.driver.task(task)
        
    def start(self):
        self.driver.start()


    def stop(self):
        self.driver.stop()
  