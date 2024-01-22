# Subt-challenge-autonomous-exploration
In this repository a simplistic autonomous exploration algotrithm is proposed to obtaine Simultaneous Localization and Mapping (SLAM) datasets in subteranean environments. Using the onboard Lidar, and depth sensors are used to with mean thresholding to detect all possible paths fron the current postition. When multiple paths is detected, exploration nodes are added to the positional mapping, to depict paths which has not yet been exploured, and aids in entire map exploration. 

# DARPA Subt Simulation Environment Installation
The simulation environment used is the [DARPA Subt](https://github.com/osrf/subt) simulation environment. Two important modifications were made to the simulation environments, the firs removing the battery management plugin, to enable endless exploration. The second being to increase the frequency of pose updates, ensuring the same frequency in pose updates than the camera images. The [Catkin System Setup](https://github.com/osrf/subt/wiki/Catkin%20System%20Setup) is follwed in installing the simulation environment with some changes.
1. Setup and install dependencies:
  ```bash
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  
  sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
  
  sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
  
  wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
  
  sudo apt-get update
  
  sudo apt-get -y install build-essential cmake libusb-dev libccd-dev libfcl-dev lsb-release pkg-config python ignition-dome ros-melodic-desktop \
  ros-melodic-tf2-sensor-msgs ros-melodic-robot-localization ros-melodic-ros-control ros-melodic-control-toolbox ros-melodic-twist-mux ros-melodic-joy \
  ros-melodic-rotors-control python3-vcstool python3-colcon-common-extensions ros-melodic-ros-ign g++-8 git python-rosdep
  ```
2. Update all ROS and Ignition packages, in case you have some of them already pre-installed:
  ```bash
  sudo apt-get upgrade
  sudo rosdep init && rosdep update
  ```
3. Set the default gcc version to 8:
  ```bash
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
  ```
4. Create a catkin workspace, and clone the SubT repository:
```bash
mkdir -p ~/subt_ws/src && cd ~/subt_ws/src
git clone https://github.com/osrf/subt
```

  ## Modifications Made to the Simulation Environment

  ## Building Procedure

# Exploration Installation

# Running Instructions
