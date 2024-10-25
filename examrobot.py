from robot import Robot

class ExamRobot(Robot):
    def __init__(self, port='/dev/ttyACM0'):
        super().__init__(port)

    def __del__(self):
        return super().__del__()
    