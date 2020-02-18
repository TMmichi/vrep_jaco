cmake_minimum_required(VERSION 3.1.3)
project(jaco_controller_integrated)

set(CMAKE_CXX_FLAGS "-std=c++0x -fpermissive ${CMAKE_CXX_FLAGS} -Wfatal-errors\ ")

find_package(catkin REQUIRED
  COMPONENTS
    action_client
    moveit_core
    moveit_visual_tools
    moveit_ros_move_group
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_ros_perception
    std_msgs
    control_msgs
    pcl_ros
    pcl_conversions
    rosbag
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

set(THIS_PACKAGE_INCLUDE_DIRS
  action_client/include/action_client
)

catkin_package(
  LIBRARIES jaco_controller_integrated
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    moveit_core
    moveit_visual_tools
    moveit_ros_move_group
    moveit_ros_planning
    moveit_ros_planning_interface
    std_msgs
    control_msgs
    tf2_geometry_msgs
    action_client
  DEPENDS
    EIGEN3
)

include_directories(include ${THIS_PACKAGE_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIRS})

add_executable(jaco_controller_integrated src/jaco_controller_integrated_jog.cpp)
target_link_libraries(jaco_controller_integrated ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_dependencies(
  jaco_controller_integrated
  ${action_client_EXPORTED_TARGETS}
)

install(TARGETS jaco_controller_integrated DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})