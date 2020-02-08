#include "jaco_controller_integrated/jaco_controller_integrated.h"

using namespace std;

double giverand(){
  mt19937_64 rng;
  // initialize the random number generator with time-dependent seed
  uint64_t timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
  seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
  rng.seed(ss);
  // initialize a uniform distribution between 0 and 1
  uniform_real_distribution<double> unif(0, 1);
  return unif(rng);
}


int main(int argc, char** argv)
{
  printf(MOVEIT_CONSOLE_COLOR_BLUE "JACO MAIN.\n" MOVEIT_CONSOLE_COLOR_RESET);
  ros::init(argc, argv, "jaco_ros_controller");
  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::NodeHandle nh;
  double p_constant;
  int p_iter;
  nh.param("/jaco_ros_controller/constant",p_constant,0.03);
  nh.param("/jaco_ros_controller/iter",p_iter,1);

  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup within controller.\n" MOVEIT_CONSOLE_COLOR_RESET);
  static const string PLANNING_GROUP_ = "arm";
  moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP_);
  printf(MOVEIT_CONSOLE_COLOR_BLUE "Move_group setup finished.\n" MOVEIT_CONSOLE_COLOR_RESET);

  const robot_state::JointModelGroup* joint_model_group =
    move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP_);
  ROS_INFO("Joint states called");

  const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();

  std::vector<double> joint_values;
  joint_values = move_group.getCurrentJointValues();
  for (std::size_t i = 0; i < joint_names.size(); ++i)
  {
    ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
  }

  ROS_INFO("End effector link: %s", move_group.getEndEffectorLink().c_str());
  
  geometry_msgs::Pose current_pose;
  current_pose = move_group.getCurrentPose().pose;
  ROS_INFO("pose_x: %f",current_pose.position.x);
  ROS_INFO("pose_y: %f",current_pose.position.y);
  ROS_INFO("pose_z: %f",current_pose.position.z);
  ROS_INFO("orientation_w: %f",current_pose.orientation.w);
  ROS_INFO("orientation_x: %f",current_pose.orientation.x);
  ROS_INFO("orientation_y: %f",current_pose.orientation.y);
  ROS_INFO("orientation_z: %f",current_pose.orientation.z);

  std::vector<geometry_msgs::Pose> waypoints;

  for (int i=0;i<p_iter;i++){
    printf(MOVEIT_CONSOLE_COLOR_BLUE "Cartesian Attempt: %d\n",i+1, MOVEIT_CONSOLE_COLOR_RESET);
    waypoints.clear();
    current_pose = move_group.getCurrentPose().pose;
    waypoints.push_back(current_pose);

    geometry_msgs::Pose target_pose = current_pose;

    target_pose.position.z += 0.01;
    target_pose.position.y += 0.01;
    target_pose.position.x += 0.01;
    waypoints.push_back(target_pose);  // down

    //target_pose.position.z += 0.02;
    //waypoints.push_back(target_pose);  //up

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    moveit_msgs::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.001;
    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

    my_plan.trajectory_ = trajectory;
    
    bool success = (move_group.execute(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    ROS_INFO_NAMED("tutorial", "Execution %s", success ? "SUCCESS" : "FAILED");
  }

  current_pose = move_group.getCurrentPose().pose;
  ROS_INFO("pose_x: %f",current_pose.position.x);
  ROS_INFO("pose_y: %f",current_pose.position.y);
  ROS_INFO("pose_z: %f",current_pose.position.z);
  ROS_INFO("orientation_w: %f",current_pose.orientation.w);
  ROS_INFO("orientation_x: %f",current_pose.orientation.x);
  ROS_INFO("orientation_y: %f",current_pose.orientation.y);
  ROS_INFO("orientation_z: %f",current_pose.orientation.z);
  //ros::waitForShutdown();
  return 0;
}