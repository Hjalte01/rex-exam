from typing import Callable
import numpy as np
from robot import Robot
from camera import Camera
from statedriver import Waitable, Driver, Event, State, Task
from pathplaning.grid import Grid, Position
from pathplaning.localization import ParticleFilter

# Driver settings
CYCLE           = 50 # ms
# Camera settings
IMG_SIZE        = (1920, 1080)
FPS             = 30
# Grid settings
LANDMARK_SIZE   = 200 # mm
ZONE_SIZE       = 450 # mm
ZONES           = 9

class ExamRobot(Waitable, Robot):
    def __init__(self, cycle=CYCLE, img_size=IMG_SIZE, fps=FPS, zone_size=ZONE_SIZE, zones=ZONES, landmark_size=LANDMARK_SIZE, port='/dev/ttyACM0'):
        # Composite inheritance requires that both supers call super(T, self).__init__(),
        # unfortunately Robot does not. Luckily Python is wonky, so this is a easy hack. 
        Waitable.__init__(self)
        Robot.__init__(self, port)
        self.driver = Driver(self, cycle)
        self.heading = 0.5 * np.pi
        self.grid = Grid((0, 0), zone_size, zones, landmark_size)
        # self.pf = ParticleFilter(Position(np.sqrt(2*(zone_size/2)**2), 0.25*np.pi), 1000, self.grid) 
        self.cam = Camera(img_size, fps, Camera.Strategy.PI_CAMERA)
        self.marker_size = landmark_size
        self.cam_matrix = None
        self.dist_coeffs = None

    def __del__(self):
        Robot.__del__(self)
    
    def capture(self):
        return self.cam.capture()
    
    def default(self, state: State):
        self.driver.default(state)

    def add(self, taskable: Task, default=False):
        self.driver.add(taskable, default)

    def switch(self, id):
        self.driver.switch(id)

    def register(self, id, listener: Callable[[Event], None]):
        self.driver.register(id, listener)
        
    def start(self):
        self.driver.start()

    def stop(self):
        Robot.stop(self)
        self.driver.stop()
  
def mock():
    class ExamRobot(Waitable):
        def __init__(self, cycle, img_size, fps, zone_size, zones, landmark_size, port='/dev/ttyACM0'):
            Waitable.__init__(self)
            self.driver = Driver(self, cycle)
            self.heading = 0.5 * np.pi
            self.grid = Grid((0, 0), zone_size, zones, landmark_size)
            # self.pf = ParticleFilter(Position(np.sqrt(2*(zone_size/2)**2), 0.25*np.pi), 1000, self.grid) 
            self.cam = Camera(img_size, fps, Camera.Strategy.TEST)
            self.marker_size = landmark_size
            self.cam_matrix = None
            self.dist_coeffs = None
        def __del__(self):
            pass
        def go_diff(self, x, y, l, r):
            pass
        def stop(self):
            self.driver.stop()
        def capture(self):
            return self.cam.capture()
        def default(self, state: State):
            self.driver.default(state)
        def add(self, taskable: Task, default=False):
            self.driver.add(taskable, default)
        def switch(self, id):
            self.driver.switch(id)
        def register(self, id, listener: Callable[[Event], None]):
            self.driver.register(id, listener)
        def wake(self):
            self.driver.wake()
            super().wake()
        def start(self):
            self.driver.start()
        def stop(self):
            self.driver.stop()

    return ExamRobot(CYCLE, IMG_SIZE, FPS, ZONE_SIZE, ZONES, LANDMARK_SIZE)
