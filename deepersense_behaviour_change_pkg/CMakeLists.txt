cmake_minimum_required(VERSION 3.0.2)
project(deepersense_behaviour_change_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  deepersense_msgs
  tf2_ros
  tf2
  tf
)

find_package(Eigen3)

catkin_package()

include_directories(
  include/deepersense_behaviour_change_pkg/
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(behaviour_change src/behaviour_change.cpp)
add_dependencies(behaviour_change ${behaviour_change_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(behaviour_change
  ${catkin_LIBRARIES}
  ${EIGEN_LIBRARIES}
)
