# Description of solution

To complete the race track, we employ a combination of Rapidly-exploring Random Trees (RRT) for path planning, a self-localization algorithm using particle filters, and an occupancy grid for mapping the environment. The solution is implemented using a state-driven architecture with threading to manage different tasks concurrently.

Key Components:
- RRT (Rapidly-exploring Random Trees):

Used for path planning to navigate the robot through the race track.
Generates a tree of possible paths from the start to the goal, ensuring efficient exploration of the space.

- Self-Localization (Particle Filter):

Uses a particle filter algorithm to estimate the robot's position and orientation.
Particles represent possible states of the robot, and their weights are updated based on sensor measurements.

- Occupancy Grid:

Represents the environment as a grid of zones.
Each zone can be marked as occupied or free, and zones can be diffused to account for obstacles.

- State-Driven Architecture:

Manages different tasks (calibration, detection, driving) using states.
Each state is handled by a separate thread, allowing concurrent execution and better performance.


# Codebase Overview
Here is a description of the different files in the codebase, focusing on their functionality and interactions.

- main.py
This is the main entry point of the application. It initializes the robot, handles user input, and manages the state transitions for different tasks such as calibration, detection, estimation, and driving.

- examrobot.py
Defines the ExamRobot class, which encapsulates the robot's hardware and software interfaces. It includes methods for capturing images, adding tasks, registering event handlers, and managing the robot's state.

- states/calibrate.py
Contains the Calibrate class and CalibrateEvent enumeration. This state handles the calibration process, including capturing images of calibration patterns and computing the camera matrix and distortion coefficients.

- states/detect.py
Contains the Detect class and DetectEvent enumeration. This state is responsible for detecting ArUco markers in the captured images and estimating their poses.

- states/drive.py
Contains the Drive class and DriveEvent enumeration. This state manages the robot's driving behavior, including navigating to specified waypoints and handling drive events.

- tasks/estimate.py
Defines the Estimate class and utility functions for pose estimation. This task uses the detected ArUco markers to estimate the robot's position and orientation in the environment.

- tasks/rrt.py
Implements the RRT (Rapidly-exploring Random Trees) algorithm for path planning. This task generates a path from the robot's current position to the goal, avoiding obstacles.

- tasks/localization.py
Implements the particle filter algorithm for self-localization. This task estimates the robot's position and orientation based on sensor data and updates the particle weights accordingly.

- tasks/occupancy_grid.py
Defines the OccupancyGrid class, which represents the environment as a grid of zones. Each zone can be marked as occupied or free, and the grid can be updated based on sensor data to account for obstacles.

# Interaction Between Files
- Initialization: 

main.py initializes the ExamRobot object and sets up the initial state.

- State Management:

main.py handles user input to switch between different states (calibration, detection, estimation, driving).
States like Calibrate, Detect, and Drive are added to the robot and managed by the state driver.

- Event Handling:

Event handlers in main.py respond to events from different states (e.g., calibration complete, detection complete) and trigger state transitions.

- Task Execution:

Tasks like Estimate, RRT, and ParticleFilter are executed within their respective states to perform specific functions (e.g., pose estimation, path planning, localization).

- Data Flow:

Sensor data (e.g., images from the camera) is captured by the ExamRobot and processed by tasks like Detect and Estimate.
Calibration data is saved and loaded from configuration files to ensure accurate pose estimation.


# Materials

-   [Event loop](https://html.spec.whatwg.org/multipage/webappapis.html#event-loops)
-   [RTT with bias](https://rllab.snu.ac.kr/courses/intelligent-systems_2016/project/project-files/instruction-for-assignment2_2016.pdf)
-   [SIR at medium](https://medium.com/@mathiasmantelli/particle-filter-part-4-pseudocode-and-python-code-052a74236ba4)
-   [SIR at Kalman & Bayesian filters](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb)
