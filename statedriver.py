"""
A very simple statedriver. It implements events, tasks & states.
The execution of a cycle is sequential: tasks -> state -> event-listeners
"""
from time import sleep
import traceback
from typing import Callable, List, Tuple
from threading import Condition, Thread
from dataclasses import dataclass

from robot import Robot

@dataclass
class EventType:
    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return self.id

class Event(object):
    """
    Event objects are passed to event handlers.
    """
    def __init__(self, type: EventType, **kwords):
        super(Event, self).__init__()
        self.type = type
        self.robot: Robot = None
        self.origin: State = None
        self.values = kwords
        for k, v in kwords.items():
            self.__setattr__(k, v)
        
    def __str__(self):
        return "{0}<\"{1}\">".format(self.__class__.__qualname__, self.type)

class Waitable(object):
    """
    A Waitable is a type that can run custom logic after a (optional) given wait.
    """
    def __init__(self):
        super(Waitable, self).__init__()
        self.__lock__ = Condition()
        self.__wait__ = True

    def __enter__(self):
        self.__lock__.acquire()
        return self.__lock__
    
    def __exit__(self, *_):
        self.__lock__.release()

    def fire(self, event: Event):
        with self:
            Driver.Events.push(event)

    def reset(self):
        with self:
            self.__wait__ = True

    def wait(self, signal: Callable[[], bool]=None):
        self.reset()
        with self as lock:
            if signal:
                lock.wait_for(signal)
                return
            while self.__wait__:
                lock.wait()

    def wait_for(self, type: EventType):
        with self as lock:
            Driver.Waiters.push((type, self))
            while self.__wait__:
                lock.wait()

    def wake(self):
        with self as lock:
            self.__wait__ = False
            lock.notify_all()

    def cancel(self):
        self.wake()


class Task(Waitable):
    """
    A Task represents a recurring task, that is run every cycle regardless of state. 
    """
    def __init__(self):
        super(Task, self).__init__()
        self.__done__ = False

    def done(self, flag=None):
        with self:
            if flag is not None:
                self.__done__ = flag
            return self.__done__

    def cancel(self):
        self.done(True)
        super().wake()

    def reset(self):
        with self:
            self.__done__ = False
        super().reset()

    def run(self, _: Robot):
        pass

class State(Task):
    """
    A state represents a unique task. Only one state is running at any given time.
    """
    def __init__(self, id: object):
        super(State, self).__init__()
        self.id = id

    def __str__(self) -> str:
        return "{0}<\"{1}\">".format(self.__class__.__qualname__, self.id)
    
class Driver(Task):
    """
    A driver handles execution of runables(Task or State objects). 
    Runables are executed sequentially in parallel with the main thread and in a threadsafe context. 
    The order of execution is **tasks -> state -> event handlers -> waitables**. 
    Exceptions from tasks, states and event handlers are caught, except for exceptions in the default state, 
    which will kill the thread gracefully. 
    """
    class Events:
        __queue__: List[Event] = []
        
        def push(e: Event):
            Driver.Events.__queue__.append(e)

    class Waiters:
        __queue__: List[Tuple[EventType, Waitable]] = []
        
        def push(val: Tuple[EventType, Waitable]):
            Driver.Waiters.__queue__.append(val)

    def __init__(self, robot: Robot, cycle=100, default_state: object = None, *states: List[State]):
        super(Driver, self).__init__()
        self.__robot                  = robot
        self.__cycle                  = cycle*0.001
        self.__states                 = dict()
        self.__active_state: State    = None
        self.__thread: Thread         = None
        self.__events                 = dict()
        self.__tasks                  = list()

        if default_state:
            self.__default = default_state
        else:
            self.__default = None

        for arg in states:
            self.add(arg)

    def __str__(self):
        return "Driver[{0}]".format(", ".join([x.__str__() for x in self.__states.values()]))

    def __len__(self):
        return len(self.__states)
        
    def __call(self, f, *args):
        with self:
            try:
                f(*args)
            except Exception:
                print("[ERR] An exception was thrown while running state {0}.".format(self.__active_state))
                traceback.print_exc()
                if self.__active_state.id == self.__default:
                    self.done(True)
                else:
                    self.switch(self.__default)

    def __runner(self):  
        while not self.done():
            # Run tasks
            for t in self.__tasks:
                if t.done(): 
                    continue
                self.__call(t.run, self.__robot)

            # Run the active state
            if self.__active_state.done():
                self.switch(self.__default)
            self.__call(self.__active_state.run, self.__robot)

            # Flush the event queue
            with self:
                while len(Driver.Events.__queue__):
                    e = Driver.Events.__queue__.pop()
                    if not e.type.id in self.__events.keys():
                        continue

                    for val in Driver.Waiters.__queue__:
                        t, w = val
                        if t.id != e.type.id:
                            continue

                        Driver.Waiters.__queue__.remove(val)
                        # Waitable.wake(w)
                        w.wake()

                    e.robot = self.__robot
                    e.origin = self.__active_state
                    for f in self.__events[e.type.id]:
                        self.__call(f, e)

            sleep(self.__cycle)
                
    def states(self):
        return (tuple(self.__states.keys()), tuple(self.__states.values()))
                
    def default(self, runable: State):
        if self.__thread:
            return
        if not runable.id in self.__states.keys():
            self.__states[runable.id] = runable
        self.__default = runable.id
        self.__active_state = self.__states[self.__default]

    def add(self, runable: Task, default=False):
        if self.__thread:
            return
        if isinstance(runable, State):
            if runable.id in self.__states.keys():
                return
            if default:
                self.default(runable)
            else:
                self.__states[runable.id] = runable
        else:
            if runable in self.__tasks:
                return
            self.__tasks.append(runable)

    def switch(self, id: object):
        if self.__active_state and self.__active_state.id == id:
            return
        if not id in self.__states.keys():
            return print("[LOG] The state \"{0}\" has not been added to driver {1}.".format(id, self))
        with self:
            self.__active_state.cancel()
            self.__active_state = self.__states[id]
            self.__active_state.reset()
        print("[LOG] Switched to state {0}.".format(self.__active_state))

    def register(self, type: EventType, handler: Callable[[Event], None]):
        with self.__lock__:
            if handler in self.__events.setdefault(type.id, []):
                return
            self.__events[type.id].append(handler)

    def unregister(self, type: EventType, handler: Callable[[Event], None]):
        with self.__lock__:
            if not (type in self.__events.keys()):
                return
            self.__events[type.id] = list(filter(lambda f: f is not handler, self.__events[type.id]))

    def start(self):
        if self.__thread:
            return
        if not self.__default:
            return print("[LOG] No default state for driver {0}. Skipping start.".format(self))
        print("[LOG] Starting driver {0}.".format(self))
        self.reset()
        self.__active_state = self.__states[self.__default]
        self.__thread = Thread(target=self.__runner, daemon=True)
        self.__thread.start()

    def stop(self):
        if self.__thread is None:
            return
        print("[LOG] Stopping driver {0}.".format(self))
        with self.__lock__:
            self.__active_state.cancel()
            self.cancel()
            self.__thread.join(10*self.__cycle)
            self.__thread = None

    def wake(self):
        self.__active_state.wake()
        super().wake()
