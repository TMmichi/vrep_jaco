#include "jaco_controller_integrated.h"


using namespace std;
using namespace jaco_controller_integrated;

JacoController::JacoController() : nh_(""), nh_local_("~"){
  updateParams();

  teleop_sub_ = nh.subscribe("key_input", 10, &JacoController::keyCallback, this);
  key_check_pub_ = nh_.advertise<std_msgs::Int8>("key_check", 10);

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup within controller.\n" MOVEIT_CONSOLE_COLOR_RESET);
  moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP_);
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  joint_model_group = move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP_);
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Joint states called\n" MOVEIT_CONSOLE_COLOR_RESET);

  current_pose = move_group.getCurrentPose().pose;

  ROS_INFO("pose_x: %f",current_pose.position.x);
  ROS_INFO("pose_y: %f",current_pose.position.y);
  ROS_INFO("pose_z: %f",current_pose.position.z);
  ROS_INFO("orientation_w: %f",current_pose.orientation.w);
  ROS_INFO("orientation_x: %f",current_pose.orientation.x);
  ROS_INFO("orientation_y: %f",current_pose.orientation.y);
  ROS_INFO("orientation_z: %f",current_pose.orientation.z);
}

void JacoController::updateParmas(){
  //nh_local_.param<double>("jaco_ros_controller/constant", p_constant, 0.01);
  //nh_local_.param<int>("jaco_ros_controller/iter", p_iter, 20);
}

void JacoController::keyCallback(const std_msgs::Int8::ConstPtr& msg){
  std_msgs::Int8 key_check.data = msg->data;
  key_check_pub_.publish(key_check);
  key_input = msg->data;

  waypoints.clear();
  current_pose = move_group.getCurrentPose().pose;
  waypoints.push_back(current_pose);
  target_pose = current_pose;

  switch(key_input){
    case 65: target_pose.position.x += 0.01;
    case 66: target_pose.position.x -= 0.01;
    case 67: target_pose.position.y += 0.01;
    case 68: target_pose.position.y -= 0.01;
    case 105: target_pose.position.z -= 0.01;
    case 111: target_pose.position.z += 0.01;
  }
  waypoints.push_back(target_pose);

  moveit_msgs::RobotTrajectory trajectory;
  fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
  my_plan.trajectory_ = trajectory;

  bool success = (move_group.execute(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  ROS_INFO_NAMED("tutorial", "Execution %s", success ? "SUCCESS" : "FAILED");
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "jaco_ros_controller");
  JacoController jc;
  ros::AsyncSpinner spinner(1);
  spinner.start();

  return 0;
}