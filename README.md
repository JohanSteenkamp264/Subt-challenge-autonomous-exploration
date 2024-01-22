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
The explourer ds1 robot is used in the recording of datasets, this is becaues of the 100m Lidar for long range path detection and the three depth cameras.

  ## Modifications Made to the Simulation Environment
  1. Removing the battery management plugin
     In the ***~/subt_ws/src/submitted_models/explorer_ds1_sensor_config_1/launch/spawner.rb*** file line 153 to 166 remove the following lines.
     ```rb
     <plugin filename="libignition-gazebo-linearbatteryplugin-system.so"
        name="ignition::gazebo::systems::LinearBatteryPlugin">
        <battery_name>linear_battery</battery_name>
        <voltage>12.694</voltage>
        <open_circuit_voltage_constant_coef>12.694</open_circuit_voltage_constant_coef>
        <open_circuit_voltage_linear_coef>-3.1424</open_circuit_voltage_linear_coef>
        <initial_charge>18.0</initial_charge>
        <capacity>18.0</capacity>
        <resistance>0.061523</resistance>
        <smooth_current_tau>1.9499</smooth_current_tau>
        <power_load>13.2</power_load>
        <start_on_motion>true</start_on_motion>
      </plugin>
     ```
  3. Increasing pose update frequency
     In the ***~/subt_ws/src/subt/submitted_models/explorer_ds1_sensor_config_1/launch/spawner.rb*** file change line 23.
     ```rb
     <static_update_frequency>1</static_update_frequency>
     ```
     to
     ```rb
     <static_update_frequency>20</static_update_frequency>
     ```
     and line 20
     ```rb
     <publish_nested_model_pose>#{$enableGroundTruth}</publish_nested_model_pose>
     ```
     to
     ```rb
     <publish_nested_model_pose>true</publish_nested_model_pose>
     ```

  ## Building Procedure
  ```bash
    # Source ROS distro's setup.bash
    source /opt/ros/melodic/setup.bash
    
    # Build and install into workspace
    cd ~/subt_ws/
    rosdep install --from-paths src --ignore-src -r
    catkin_make -DCMAKE_BUILD_TYPE=Release install
  ```

## IGN Download Procedure Before First Launch
  Because of some problems encountered when dounloading during launching the environment the first time. This command does take a lot of time, especially with the ***-j 1** stipulating one thread. This is due to errors encountered when using more than one.
  ```bash
    ign fuel download -v 4 -j 1 -u "https://fuel.ignitionrobotics.org/OpenRobotics/collections/SubT Tech Repo"
  ```
  
# Exploration Installation
  ```bash
    mkdir ~/catkin_ws
    mkdir ~/catkin_ws/src
    cd ~/catkin_ws/src
    
    git clone https://github.com/JohanSteenkamp264/Subt-challenge-autonomous-exploration.git
    
    cd ~/catkin_ws/src/Subt-challenge-autonomous-exploration/Scripts
    chmod +x *.py
    
    cd ~/catkin_ws/
    
    # Source ROS distro's setup.bash
    source /opt/ros/melodic/setup.bash
    
    rosdep install --from-paths src --ignore-src -r
    catkin_make
    
  ```
    
# Running Instructions

  ## Starting the Simulation Environments
  ```bash
    source /opt/ros/melodic/setup.bash
    source ~/subt_ws/install/setup.bash 
    ign launch -v 4 competition.ign worldName:=cave_circuit_01 circuit:=cave enableGroundTruth:=true robotName1:=X1 robotConfig1:=EXPLORER_DS1_SENSOR_CONFIG_1
  ```

  ## Starting Exploration Algorithm
  ```bash
    source /opt/ros/melodic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    rosrun Subt-challenge-autonomous-exploration exploration.py #{saving path} #{maximum velocity}
  ```

  ## Limiting Dataset Length
  In order to limit the length in dataset to a number of frames, before a turn around point is searchd for the following two lines in the ***exploration.py*** controls this functionality.
  ```bash
    self.limit_Dataset_genetation = True
    self.limit_index = 6000
  ```
  
