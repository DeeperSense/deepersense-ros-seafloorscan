cmake_minimum_required(VERSION 3.0.2)
project(deepersense_classify_segment_pkg)

find_package(catkin REQUIRED COMPONENTS
  deepersense_msgs
  rospy
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
