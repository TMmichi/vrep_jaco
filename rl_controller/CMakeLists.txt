cmake_minimum_required(VERSION 2.8.3)
project(rl_controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  tf
  message_generation
  std_msgs
  geometry_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  InitTraining.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(CATKIN_DEPENDS
  rospy
  tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/rl_controller_script.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
