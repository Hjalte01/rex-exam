from time import sleep
from statedriver import Event, State
from examrobot import ExamRobot

class Detect(State):
    def __init__(self):
        super().__init__("STATE_DETECT")
        self.first = None
        self.moving = False
    
    def run(self, robot: ExamRobot):
        if not self.moving:
            self.moving = True
            robot.go_diff(40, 40, 1, 0)
            return
        
        robot.stop()
        sleep(0.1)
        
