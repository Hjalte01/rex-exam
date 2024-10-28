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

    def __init__(self, img_size: Tuple[int, int], fps: int, strategy: int):
        super(Camera, self).__init__()

        self.strategy = strategy
        self.img_size = img_size
        if  strategy == Camera.Strategy.PI_CAMERA \
            or strategy == Camera.Strategy.PI_CAMERA_REQ:
                self.__cam__ = picamera2.Picamera2()
                self.config = self.cam.create_video_configuration({
                    "size": img_size, 
                    "format": "RGB888"
                    },
                    controls={
                        "FrameDurationLimits": (int(1/fps * 1000000), int(1/fps * 1000000)),
                    },
                    queue=False
                )
                self.__cam__.configure(self.config)
                self.__cam__.start(show_preview=False)
        else:
            # GStream configuration here.
            pass
        self.__task__ = CaptureTask(self.__cam__)

    def capture(self):
        if self.strategy == Camera.Strategy.PI_CAMERA:
            return self.cam.capture_array("main")
        elif self.cam_strategy == Camera.Strategy.PI_CAMERA_REQ:
            return self.__task__.get()
        else:
            # GStream capture here.
            # Make sure libcamera is installed: gst-inspect-1.0 libcamerasrc
            # Try without videobox and/or appsink
            # Force BGR: video/x-raw, format=BGR
            pass

    def stop(self):
        """
        Stops the camera.
        """
        try:
            self.__cam__.stop()
        except:
            pass

class CaptureTask(Task):
    def __init__(self, cam: picamera2.Picamera2):
        super().__init__()
        self.__cam__ = cam
        self.__frame__ = None

    def __signal__(self, task):
        req = self.__frame__.wait(task)
        self.__frame__ = req.make_array("main")
        self.wake()
        req.release()

    def run(self):
        with self:
            self.__cam__.capture_request(signal_function=self.__signal__)

    def get(self):
        with self:
            self.run()
            self.wait()
            return self.__frame__
