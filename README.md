# Sea-Floor Scan ROS System

This repository includes all the ROS packages implemented for the seafloorscan inference. It listens to the side-scan sonar and AUV navigation data and performs real-time side-scan sonar data preprocessing, undistortion and sea-floor classification and segmentation. 

## Prerequisites 

- ROS
- NVIDIA driver 
- TensorFlow 
- PyTorch
- TorchVision 
- PyXTF 
- OpenCV

## Description 

### deepersense_main_pkg

Runs *deepersense_waterfall_gen_pkg*, *deepersense_classify_segment_pkg* and *deepersense_behaviour_change_pkg*.

### deepersense_msgs

Custom ROS messages.

### deepersense_sonar_simulator_pkg

Reads .xtf and .bag file data and publishes it into the topics for the *deepersense_waterfall_gen_pkg*.

### deepersense_waterfall_gen_pkg

Receives side-scan sonar and navigation data, creates the waterfall images, corrects them and re-publishes them for the *deepersense_classify_segment_pkg*.

### deepersense_classify_segment_pkg

Receives waterfall data and performs classification and segmentation and publishes it for the *deepersense_behaviour_change_pkg*. 

### deepersense_behaviour_change_pkg

Receives the inference output and performs simulated behavioural changes to the robot.

### deepersense_vis_pkg

Runs a customised Rviz configuration for the project.

### deepersense_record_pkg

Records specific topics to replay the mission at a later time.

## Installation

Clone repository in catkin workspace.
```
cd ~/catkin_ws/src
git clone `https://github.com/DeeperSense/deepersense-ros-seafloorscan`
```

First build the custom messages.
```
cd ..
catkin build deepersense_msgs
source devel/setup.bash 
```

Then build the rest of the packages.
```
catkin build
source devel/setup.bash
```

## Usage

There are two input modalities:

1. **XTF only**
    - Set the parameter **/simulator/combined_data** to **True**. 
    - Place the .xtf file inside the **deepersense_sonar_simulator_pkg/cfg/sonar** folder. 
    
Only [.xtf](https://www.ecagroup.com/en/xtf-file-format) file input containing both ping intensity and navigation information, where the navigation data must be in UTM coordinates. 

2. **XTF + AUV navigation**
    - Set the parameter **/simulator/combined_data** to **False**. 
    - Place the .xtf file inside the **deepersense_sonar_simulator_pkg/cfg/sonar** folder.
    - Place the .bag file inside the **deepersense_sonar_simulator_pkg/cfg/nav** folder. 
    
Ping intensity and navigation information split between an .xtf and a [.bag](http://wiki.ros.org/Bags) file. *Remember* that the timestamps in these two files need to be synchronised for the system to work. The navigation topic must be of [**cola2_msgs/NavSts**](https://bitbucket.org/iquarobotics/cola2_msgs/src/master/msg/) type. Internally, the system will produce a combined .bag file (saved inside **deepersense_sonar_simulator_pkg/cfg/merged**) that will be used the next time the program is run with the same setup to avoid repeating the timestamp-matching process. 

### Select packages to run
Inside **deepersense_main_pkg/launch/main.yaml**:
- Set *classify* if you want to run *deepersense_classify_segment_pkg*
- Set *behaviour* if you want to run *deepersense_behaviour_change_pkg*

### run system 

```
roslaunch deepersense_main_pkg main.launch
```

### run simulator

```
roslaunch deepersense_sonar_simulator_pkg simulate.launch
```

### run visualisation
```
roslaunch deepersense_vis_pkg visualise.launch
```

### record 
Modify **deepersense_record_pkg/cfg/topics.txt** and select the topics that you would like to record on a rosbag. 
```
roslaunch deepersense_record_pkg record.launch
```

## replay 
```
rosbag play <bag file>
roslaunch deepersense_vis_pkg visualise.launch
```

They can be run from the same computer or from different ones provided they are all in the same ROS local network. 


## ROS Parameters Description

Modifiable parameters inside **/deepersense_main_pkg/cfg/pipeline_config.yaml**:

Sonar parameters
- **/sonar/num_samples**: side-scan sonar number of samples in .xtf file
- **/sonar/slant_range**: side-scan sonar slant-range in .xtf file

Simulator parameters
- **/simulator/separate_inputs**: modality 1 or 2
- **/simulator/publish_rate:** xtf publish rate when modality 1 selected
- **/simulator/nav_topic_name** : navigation topic name in .bag when modality 2 selected
- **/simulator/pings_topic_name**: pings topic name 
- **/simulator/ping_info_topic_name**: ping info topic name 
- **/simulator/xtf_file**: scan sonar .xtf file name 
- **/simulator/ping_start** : ping index to start reading the .xtf file from
- **/simulator/nav_rosbag_file**: navigation .bag file name 
- **/simulator/robot_dae_file**: optional .dae file for the robot visualisation on rviz 
- **/simulator/robot_mesh_rotation_offset**: fixed rotational offset (degrees) of mesh  

Waterfall correction parameters 
- **/undistortion/correct_waterfall**: perform waterfall undistortion using navigation data 
- **/undistortion/max_pings**: number of pings to use for undistortion and inference
- **/undistortion/interpolation_window**: number of pings between previous inference and the next 

Behaviour parameters
- **/behaviour/num_steps** : number of steps to visualise the robot change of behaviour on rviz 
- **/behaviour/step_rate**: number of steps per second (Hz)
- **/behaviour/slow_down_rate**: slow down rate wrt current velocity  
- **/behaviour/min_confidence**: minimum average confidence, triggers stop behaviour

Prediction parameters
- **/prediction/patch_shape**: patch shape to extract from waterfall for inference (smaller or equal to *max_pings*)
- **/prediction/stride**: stride to extract patches from waterfall 
- **/prediction/encoder**: encoder type
- **/prediction/decoder**: decoder type

Visualisation parameters
- **/visualisation/resolution**: grid-cells resolution visualisation on rviz 
- **/visualisation/publish_prediction**: publish model inference prediction on Rviz
- **/visualisation/publish_grid_cells**: publish grid cells on Rviz
- **/visualisation/publish_swath_lines**: publish swath lines on Rviz
- **/visualisation/publish_undistortion**: publish undistorted waterfall on Rviz
- **/visualisation/publish_waterfall**: publish distorted waterfall on Rviz

