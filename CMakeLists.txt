cmake_minimum_required(VERSION 2.8.3)
project(sem_keypts_dataset_generator)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp rospy
  sensor_msgs std_msgs
  geometry_msgs tf
  cv_bridge image_transport
  eigen_conversions random_numbers
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 3.2 REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy tf
      cv_bridge eigen_conversions
      image_transport std_msgs
      geometry_msgs sensor_msgs
      random_numbers
  DEPENDS EIGEN3 OpenCV)

###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/image_processor_node.cpp src/image_processor.cpp)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
