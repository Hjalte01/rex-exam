from statedriver import Task
from examrobot import ExamRobot

class Correct(Task):
    def __init__(self):
        super(Correct, self).__init__()

    def run(self, robot: ExamRobot):
        pass
