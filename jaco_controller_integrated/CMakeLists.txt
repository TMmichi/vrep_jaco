cmake_minimum_required(VERSION 3.1.3)
project(jaco_controller_integrated)

set(CMAKE_CXX_FLAGS "-std=c++0x -fpermissive ${CMAKE_CXX_FLAGS} -Wfatal-errors\ ")

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  trajectory_msgs
  actionlib
  moveit_ros_planning_interface
)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES jaco_controller_integrated
  CATKIN_DEPENDS
    std_msgs
    geometry_msgs
    trajectory_msgs
    actionlib
    moveit_ros_planning_interface
  DEPENDS
    EIGEN3
)

include_directories(
  include 
  ${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIRS}
)

add_library(jaco_controller_integrated SHARED
  ## src/jaco_controller_integrated_jog.cpp
  src/jaco_controller_integrated_jog.cpp
  src/low_pass_filter.cpp
)
add_dependencies(jaco_controller_integrated ${catkin_EXPORTED_TARGETS})
target_link_libraries(jaco_controller_integrated
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)

add_executable(jaco_controller_integrated_jog
  ## src/jaco_controller_integrated.cpp
  src/jaco_controller_integrated_jog.cpp
)

##################
## INSTALLATION ##
##################

install(TARGETS jaco_controller_integrated DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})