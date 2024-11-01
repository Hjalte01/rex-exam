from typing import List
import numpy as np
from examrobot import ExamRobot
from statedriver import Event, EventType, State
from pathplaning.rrt import rrt_path
from pathplaning.grid import Cell

class DriveEvent(Event):
    NODE_VISITED = EventType("EVENT-DRIVE-NODE-VISITED")
    GOAL_VISITED = EventType("EVENT-DRIVE-GOAL-VISITED")
    COMPLETE = EventType("EVENT_DRIVE_COMPLETE")

    def __init__(self, type, **kwords):
        super().__init__(type, **kwords)
        self.robot: ExamRobot = None

class Drive(State):
    ID = "STATE_DRIVE"

    def __init__(self, ids: List[int], perimeter = 400):
        super(Drive, self).__init__(Drive.ID)
        self.ids = ids
        self.perimeter = perimeter
        self.moving = False
        self.path: List[Cell] = None
        self.target: Cell = None
        
    def run(self, robot: ExamRobot):
        if not self.moving:
            next = self.ids.pop()
            
            for m in robot.grid.markers:
                if m.id != next:
                    continue
                self.target = m
                break

            if self.target is None:
                self.done(True)
                return
            
            self.path = rrt_path(robot.grid, robot.grid.origo, self.target)
            self.target = self.path.pop()
            self.moving = True

        dx, dy = self.target.cx - robot.grid.origo.cx, self.target.cy - robot.grid.origo.cy
        theta = np.arctan2(dy, dx)
        offset = np.pi*2*0.05


        if theta - offset < robot.heading:
            robot.go_diff(40, 40, 0, 1)
            robot.log_file.write("[LOG] {0} - Rotating left to {1} with heading {2}.".format(self, theta, robot.heading))
            return
        elif theta + offset > robot.heading:
            robot.go_diff(40, 40, 1, 0)
            robot.log_file.write("[LOG] {0} - Rotating right to {1} with heading {2}.".format(self, theta, robot.heading))
            return
        else:
            robot.go_diff(40, 40, 1, 1)
            robot.log_file.write("[LOG] {0} - Driving to {1}.".format(self, self.target))


        if np.sqrt(dx**2 + dy**2) >= self.perimeter:
            return

        if len(self.path):
            robot.log_file.write("[LOG] {0} - Visited cell {1}.".format(self, self.target))
            self.fire(DriveEvent(DriveEvent.NODE_VISITED, self.target))
            self.target = self.path.pop()
            return
        else:
            robot.log_file.write("[LOG] {0} - Visited goal {1}.".format(self, self.target))
            robot.go_diff(0, 0, 1, 1)
            self.moving = False
            self.fire(DriveEvent(DriveEvent.GOAL_VISITED, self.target))
        
        if not len(self.ids) and not len(self.path):
            robot.log_file.write("[LOG] {0} - Visited all goals.".format(self))
            robot.go_diff(0, 0, 1, 1)
            self.fire(DriveEvent(DriveEvent.COMPLETE))
