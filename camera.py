import cv2
from dataclasses import dataclass
from typing import Tuple
from statedriver import Task

try:
    import picamera2  # type: ignore
except:
    pass

class Camera(object):
    """
    Camera objects exposes a standarized API for capturing a frame.
    """
    @dataclass
    class Strategy:
        PI_CAMERA       = 0
        PI_CAMERA_REQ   = 1
        GSTREAM         = 2
        TEST            = 3

    def __init__(self, img_size: Tuple[int, int], fps: int, strategy: int):
        super(Camera, self).__init__()

        self.strategy = strategy
        self.img_size = img_size
        if strategy == Camera.Strategy.PI_CAMERA \
            or strategy == Camera.Strategy.PI_CAMERA_REQ:
                self.__cam = picamera2.Picamera2()
                self.config = self.__cam.create_video_configuration({
                    "size": img_size, 
                    "format": "RGB888"
                    },
                    controls={
                        "FrameDurationLimits": (int(1/fps * 1000000), int(1/fps * 1000000)),
                    },
                    queue=False
                )
                self.__cam.configure(self.config)
                self.__cam.start(show_preview=False)
        elif strategy == Camera.Strategy.GSTREAM:
            # GStream configuration here.
            pass
        else:
            self.__cam = cv2.VideoCapture(0)

        self.__task__ = CaptureTask(self.__cam)

    def capture(self):
        if self.strategy == Camera.Strategy.PI_CAMERA:
            return self.cam.capture_array("main")
        elif self.strategy == Camera.Strategy.PI_CAMERA_REQ:
            return self.__task__.get()
        elif self.strategy == Camera.Strategy.GSTREAM:
            # GStream capture here.
            # Make sure libcamera is installed: gst-inspect-1.0 libcamerasrc
            # Try without videobox and/or appsink
            # Force BGR: video/x-raw, format=BGR
            pass
        else:
            _, frame = self.__cam.read()
            return frame

    def stop(self):
        """
        Stops the camera.
        """
        try:
            self.__cam.stop()
        except:
            pass

class CaptureTask(Task):
    def __init__(self, cam):
        super().__init__()
        self.__cam = cam
        self.__frame = None

    def __signal__(self, task):
        req = self.__frame.wait(task)
        self.__frame = req.make_array("main")
        self.wake()
        req.release()

    def run(self):
        with self:
            self.__cam.capture_request(signal_function=self.__signal__)

    def get(self):
        with self:
            self.run()
            self.wait()
            return self.__frame
