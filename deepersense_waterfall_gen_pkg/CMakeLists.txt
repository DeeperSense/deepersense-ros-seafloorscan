cmake_minimum_required(VERSION 3.0.2)
project(deepersense_waterfall_gen_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  deepersense_msgs
  cola2_msgs
  cv_bridge
  tf2
  tf2_ros
)

find_package(Eigen3)
find_package(CUDA REQUIRED)
find_package(PythonLibs 2.7)

# Set the CUDA architecture (change it according to your GPU)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -rdc=true -w -O3")

# only necessary if you wish to create a lib
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS OFF)

# Set the C++ standard to C++11
set(CMAKE_CXX_STANDARD 11)

catkin_package()

include_directories(
  include/deepersense_waterfall_gen_pkg
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS} 
  ${PYTHON_INCLUDE_DIRS}
)

cuda_add_library(waterfall_utils ${LIB_TYPE}
  cuda/waterfall_utils.cu
  ${INCLUDE_DIRECTORIES}
)

cuda_add_library(waterfall_gen
  cuda/waterfall_gen.cu
  ${INCLUDE_DIRECTORIES}
)

cuda_add_executable(waterfall_ros_node cuda/waterfall_ros_node.cu)

add_dependencies(waterfall_ros_node ${waterfall_ros_node_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} deepersense_msgs_generate_messages_cpp)

target_link_libraries(waterfall_ros_node
  waterfall_gen
  waterfall_utils
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PYTHON_LIBRARIES}
)